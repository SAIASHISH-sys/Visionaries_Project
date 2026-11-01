from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from fpgraph import load_shard, get_tile, NODE_FEATURES, TARGET_CLASSES

TILE_MM = 10000.0
NUM_CLASSES = len(TARGET_CLASSES)
F_COS2T, F_SIN2T = NODE_FEATURES.index("cos2theta"), NODE_FEATURES.index("sin2theta")
F_CX, F_CY = NODE_FEATURES.index("cx_tile"), NODE_FEATURES.index("cy_tile")


def load_tiles(root, split, src, max_shards=None):
    files = sorted((Path(root) / "graphs" / split).glob("*.npz"))
    if max_shards:
        files = files[:max_shards]
    tiles = []
    for f in files:
        shard = load_shard(f)
        for k in range(len(shard["node_ptr"]) - 1):
            t = get_tile(shard, k)
            t["src"] = src
            tiles.append(t)
    return tiles


def class_stats(tiles, num_classes=NUM_CLASSES):
    counts = np.zeros(num_classes, dtype=np.int64)
    pos = masked = 0
    for t in tiles:
        counts += np.bincount(t["y"].astype(np.int64), minlength=num_classes)
        pos += int(t["edge_y"].sum())
        masked += int(t["edge_mask"].sum())
    freq = counts / max(counts.sum(), 1)
    present = counts > 0
    med = np.median(freq[present]) if present.any() else 1.0
    weights = np.where(present, np.clip(med / np.maximum(freq, 1e-12), 0.1, 10.0), 0.0)
    return {"counts": counts, "weights": weights, "pos_weight": float((masked - pos) / max(pos, 1))}


def blend_stats(stats_list):
    w = np.stack([s["weights"] for s in stats_list])
    present = w > 0
    weights = np.where(present.any(0), (w * present).sum(0) / np.maximum(present.sum(0), 1), 0.0)
    return {"counts": sum(s["counts"] for s in stats_list), "weights": weights,
            "pos_weight": float(np.mean([s["pos_weight"] for s in stats_list]))}


def transform_pts(pts, k, mirror, tile_size=TILE_MM):
    p = pts.astype(np.float64) - tile_size / 2
    if mirror:
        p[..., 0] = -p[..., 0]
    for _ in range(k % 4):
        p = np.stack([-p[..., 1], p[..., 0]], -1)
    return (p + tile_size / 2).astype(np.float32)


def recompute_node_features(x, pts, ptype, tile_size=TILE_MM):
    out = x.copy()
    p = pts.astype(np.float64)
    d = p[:, 3] - p[:, 0]
    th = np.arctan2(d[:, 1], d[:, 0])
    mid = p.mean(1)
    oriented = ~np.isin(ptype, (2, 3))
    out[oriented, F_COS2T] = np.cos(2 * th)[oriented]
    out[oriented, F_SIN2T] = np.sin(2 * th)[oriented]
    out[:, F_CX] = mid[:, 0] / tile_size
    out[:, F_CY] = mid[:, 1] / tile_size
    return out


def node_length_mm(x):
    return np.expm1(x[:, NODE_FEATURES.index("log_length_m")].astype(np.float64)) * 1000.0


def collate(tiles, transforms=None):
    xs, eis, eas, ys, insts, eys, ems, bs, srcs = ([] for _ in range(9))
    node_ptr, edge_ptr = [0], [0]
    n_off = inst_off = 0
    for b, t in enumerate(tiles):
        x = t["x"]
        if transforms is not None and (transforms[b][0] % 4 or transforms[b][1]):
            k, mirror = transforms[b]
            x = recompute_node_features(x, transform_pts(t["pts"], k, mirror), t["ptype"])
        n = len(t["y"])
        inst = t["inst"].astype(np.int64)
        xs.append(x)
        eis.append(t["edge_index"].astype(np.int64) + n_off)
        eas.append(t["edge_attr"])
        ys.append(t["y"].astype(np.int64))
        insts.append(np.where(inst >= 0, inst + inst_off, -1))
        eys.append(t["edge_y"].astype(np.float32))
        ems.append(t["edge_mask"].astype(bool))
        bs.append(np.full(n, b, dtype=np.int64))
        srcs.append(np.full(n, t.get("src", 0), dtype=np.int64))
        if n and inst.max() >= 0:
            inst_off += int(inst.max()) + 1
        n_off += n
        node_ptr.append(n_off)
        edge_ptr.append(edge_ptr[-1] + t["edge_index"].shape[1])
    cat = lambda a, ax=0: torch.from_numpy(np.concatenate(a, axis=ax))
    return {
        "x": cat(xs), "edge_index": cat(eis, 1), "edge_attr": cat(eas), "y": cat(ys), "inst": cat(insts),
        "edge_y": cat(eys), "edge_mask": cat(ems), "batch": cat(bs), "node_src": cat(srcs),
        "node_ptr": torch.tensor(node_ptr), "edge_ptr": torch.tensor(edge_ptr),
    }


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def make_batches(sizes, idx, node_budget, max_tiles, rng=None):
    order = np.asarray(idx)
    if rng is not None:
        order = order[rng.permutation(len(order))]
    batches, cur, n = [], [], 0
    for i in order:
        s = int(sizes[i])
        if cur and (n + s > node_budget or len(cur) >= max_tiles):
            batches.append(cur)
            cur, n = [], 0
        cur.append(int(i))
        n += s
    if cur:
        batches.append(cur)
    return batches


def pad_shard(batches, rank, world):
    if not batches:
        return []
    pad = (-len(batches)) % world
    batches = batches + [batches[i % len(batches)] for i in range(pad)]
    return batches[rank::world]


class BatchDataset(Dataset):
    def __init__(self, tiles, batches, augment=False, seed=0, epoch=0):
        self.tiles, self.batches, self.augment, self.seed, self.epoch = tiles, batches, augment, seed, epoch

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        group = [self.tiles[j] for j in self.batches[i]]
        transforms = None
        if self.augment:
            rng = np.random.default_rng([self.seed, self.epoch, int(self.batches[i][0])])
            transforms = [(int(rng.integers(4)), bool(rng.integers(2))) for _ in group]
        return collate(group, transforms)
