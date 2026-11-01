import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fpgraph import TARGET_CLASSES, NODE_FEATURES, EDGE_FEATURES

NUM_CLASSES = len(TARGET_CLASSES)
LEN_RATIO = EDGE_FEATURES.index("log_len_ratio")
OTHER_ID = TARGET_CLASSES.index("other")
ELEVATOR_ID = TARGET_CLASSES.index("elevator")


@dataclass
class ModelConfig:
    node_dim: int = len(NODE_FEATURES)
    edge_dim: int = len(EDGE_FEATURES)
    dim: int = 128
    heads: int = 8
    stages: int = 8
    edge_emb: int = 64
    ffn_mult: int = 2
    dropout: float = 0.0
    num_classes: int = NUM_CLASSES

    def to_dict(self):
        return asdict(self)


def scatter_softmax(scores, index, n):
    h = scores.shape[1]
    idx = index[:, None].expand(-1, h)
    top = torch.full((n, h), float("-inf"), dtype=scores.dtype, device=scores.device)
    top = top.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
    ex = torch.exp(scores - top[index])
    den = torch.zeros((n, h), dtype=scores.dtype, device=scores.device).index_add_(0, index, ex)
    return ex / (den[index] + 1e-16)


def reverse_edge_attr(edge_attr):
    out = edge_attr.clone()
    out[:, LEN_RATIO] = -out[:, LEN_RATIO]
    return out


class SparseGATStage(nn.Module):
    def __init__(self, dim, heads, edge_emb, ffn_mult, dropout):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh = heads, dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.edge_bias = nn.Linear(edge_emb, heads)
        self.self_bias = nn.Parameter(torch.zeros(heads))
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_mult * dim), nn.GELU(), nn.Linear(ffn_mult * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src, dst, emb):
        n = x.shape[0]
        q, k, v = self.qkv(self.norm1(x)).view(n, 3, self.h, self.dh).unbind(1)
        loops = torch.arange(n, device=x.device)
        s_all, d_all = torch.cat([src, loops]), torch.cat([dst, loops])
        score = (q[d_all] * k[s_all]).sum(-1) / math.sqrt(self.dh)
        bias = torch.cat([self.edge_bias(emb), self.self_bias.to(score.dtype).expand(n, -1)])
        alpha = scatter_softmax((score + bias).float(), d_all, n)
        msg = v[s_all].float() * alpha[..., None]
        agg = torch.zeros(n, self.h, self.dh, dtype=torch.float32, device=x.device).index_add_(0, d_all, msg)
        x = x + self.drop(self.out(agg.reshape(n, -1))).float()
        x = x + self.drop(self.ffn(self.norm2(x))).float()
        return x, alpha[:src.shape[0]]


class FloorGAT(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        self.node_in = nn.Sequential(nn.Linear(c.node_dim, c.dim), nn.GELU(), nn.Linear(c.dim, c.dim))
        self.rse = nn.Sequential(nn.Linear(c.edge_dim, c.edge_emb), nn.GELU(), nn.Linear(c.edge_emb, c.edge_emb))
        self.stages = nn.ModuleList(
            [SparseGATStage(c.dim, c.heads, c.edge_emb, c.ffn_mult, c.dropout) for _ in range(c.stages)])
        self.final_norm = nn.LayerNorm(c.dim)
        self.head1 = nn.Sequential(nn.Linear(c.dim, c.dim), nn.GELU(), nn.Linear(c.dim, c.num_classes))
        self.node_to_edge = nn.Linear(c.dim, c.edge_emb)
        cee_in = c.stages * c.heads + 3 * c.edge_emb
        self.head2 = nn.Sequential(nn.LayerNorm(cee_in), nn.Linear(cee_in, 128), nn.GELU(),
                                   nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        for last in (self.head1[-1], self.head2[-1]):
            nn.init.normal_(last.weight, std=0.01)
            nn.init.zeros_(last.bias)

    def forward(self, x, edge_index, edge_attr, return_attn=False):
        e = edge_index.shape[1]
        src = torch.cat([edge_index[0], edge_index[1]])
        dst = torch.cat([edge_index[1], edge_index[0]])
        emb = self.rse(torch.cat([edge_attr, reverse_edge_attr(edge_attr)]))
        h = self.node_in(x).float()
        attn = []
        for stage in self.stages:
            h, a = stage(h, src, dst, emb)
            attn.append((a[:e] + a[e:]) / 2)
        h = self.final_norm(h)
        node_logits = self.head1(h).float()
        u = self.node_to_edge(h).float()
        i, j = edge_index
        sym = (emb[:e].float() + emb[e:].float()) / 2
        feat = torch.cat([torch.cat(attn, 1), u[i] + u[j], (u[i] - u[j]).abs(), sym], 1)
        out = {"node_logits": node_logits, "edge_logits": self.head2(feat).squeeze(-1).float()}
        if return_attn:
            out["attn"] = torch.stack(attn, 1)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PanopticLoss(nn.Module):
    def __init__(self, class_weights, edge_pos_weight, lam=2.0, gamma=0.0):
        super().__init__()
        self.register_buffer("cw", torch.as_tensor(class_weights, dtype=torch.float32))
        self.register_buffer("pw", torch.tensor(float(edge_pos_weight)))
        self.lam, self.gamma = lam, gamma

    def forward(self, out, y, edge_y, edge_mask, node_src=None):
        logits = out["node_logits"].float()
        ce = F.cross_entropy(logits, y, reduction="none")
        if self.gamma > 0:
            ce = ce * (1 - torch.exp(-ce)).pow(self.gamma)
        w = self.cw[y]
        if node_src is not None:
            forgive = (node_src == 1) & (y == OTHER_ID) & (logits.argmax(1) == ELEVATOR_ID)
            w = torch.where(forgive, torch.zeros_like(w), w)
        node = (ce * w).sum() / w.sum().clamp(min=1e-6)
        bce = F.binary_cross_entropy_with_logits(out["edge_logits"].float(), edge_y.float(), pos_weight=self.pw,
                                                 reduction="none")
        m = edge_mask.float()
        edge = (bce * m).sum() / m.sum().clamp(min=1.0)
        total = node + self.lam * edge
        return total, {"total": total.detach(), "node": node.detach(), "edge": edge.detach()}
