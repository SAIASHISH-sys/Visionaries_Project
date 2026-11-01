import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from gnnmodel import FloorGAT, ModelConfig, PanopticLoss, count_parameters
from gnndata import load_tiles, class_stats, blend_stats, make_batches, pad_shard, BatchDataset, to_device
from gnneval import Evaluator, evaluate, DEFAULT_THRESHOLDS


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--fp-dir", required=True)
    p.add_argument("--ind-dir", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["pretrain", "finetune"], default="pretrain")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-step", type=int, default=20)
    p.add_argument("--lr-gamma", type=float, default=0.7)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=2.0)
    p.add_argument("--focal-gamma", type=float, default=0.0)
    p.add_argument("--thr", type=float, default=0.7)
    p.add_argument("--node-budget", type=int, default=12000)
    p.add_argument("--max-tiles", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--stages", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--val-set", choices=["fp", "ind"], default="fp")
    p.add_argument("--aux-val", choices=["", "fp", "ind"], default="")
    p.add_argument("--max-fp-shards", type=int, default=0)
    p.add_argument("--max-val-tiles", type=int, default=0)
    p.add_argument("--max-train-tiles", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--init", default="")
    p.add_argument("--freeze-stages", type=int, default=0)
    p.add_argument("--freeze-epochs", type=int, default=0)
    p.add_argument("--fp-mix", type=float, default=0.25)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", default="")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args(argv)


def setup_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    cuda = torch.cuda.is_available()
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    if cuda:
        torch.cuda.set_device(local)
    return rank, world, local, torch.device("cuda", local) if cuda else torch.device("cpu")


def reduce_sum(values, device):
    t = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t)
    return t.cpu().numpy()


def reduce_max(value, device):
    t = torch.tensor([value], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def make_loader(tiles, batches, args, augment, epoch):
    ds = BatchDataset(tiles, batches, augment=augment, seed=args.seed, epoch=epoch)
    w = args.workers
    return DataLoader(ds, batch_size=None, shuffle=False, num_workers=w, prefetch_factor=2 if w > 0 else None)


def subsample(tiles, n, seed):
    if n and len(tiles) > n:
        keep = np.random.default_rng(seed).choice(len(tiles), n, replace=False)
        return [tiles[i] for i in sorted(keep)]
    return tiles


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def lr_for(group, epoch, step, args):
    if epoch < group["frozen_until"]:
        return 0.0
    base = group["base_lr"] * args.lr_gamma ** (epoch // args.lr_step)
    return base * min(1.0, (step + 1) / max(args.warmup_steps, 1))


def run_validation(model, tiles, loss_fn, args, rank, world, device):
    sizes = np.array([len(t["y"]) for t in tiles])
    batches = make_batches(sizes, np.arange(len(tiles)), args.node_budget, args.max_tiles)[rank::world]
    loader = make_loader(tiles, batches, args, False, 0)
    ev = evaluate(model, loader, loss_fn, device, DEFAULT_THRESHOLDS, amp=args.amp and device.type == "cuda")
    total = Evaluator(DEFAULT_THRESHOLDS)
    total.load(reduce_sum(ev.vector(), device))
    return total.summarize(args.thr)


def select_score(summary):
    return summary["pq"] if summary["n_gt_instances"] > 0 and summary["pq"] == summary["pq"] else summary["wf1"]


def main(argv=None):
    args = parse_args(argv)
    rank, world, local, device = setup_dist()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    amp = args.amp and device.type == "cuda"
    finetune = args.mode == "finetune"

    fp_train = load_tiles(args.fp_dir, "train", 0, args.max_fp_shards or None)
    ind_train = load_tiles(args.ind_dir, "train", 1) if finetune else []
    if finetune:
        pool = ind_train + fp_train
    else:
        pool = fp_train
    pool = subsample(pool, args.max_train_tiles, args.seed)
    n_ind = sum(1 for t in pool if t["src"] == 1)
    val_dirs = {"fp": (args.fp_dir, 0), "ind": (args.ind_dir, 1)}
    val_sets = {}
    for name in [args.val_set] + ([args.aux_val] if args.aux_val and args.aux_val != args.val_set else []):
        d, s = val_dirs[name]
        shards = (args.max_fp_shards or None) if name == "fp" else None
        val_sets[name] = subsample(load_tiles(d, "val", s, shards), args.max_val_tiles, args.seed)
    if finetune and n_ind:
        stats = blend_stats([class_stats([t for t in pool if t["src"] == 1]),
                             class_stats([t for t in pool if t["src"] == 0])])
    else:
        stats = class_stats(pool)
    sizes = np.array([len(t["y"]) for t in pool])
    ind_idx = np.array([i for i, t in enumerate(pool) if t["src"] == 1], dtype=int)
    fp_idx = np.array([i for i, t in enumerate(pool) if t["src"] == 0], dtype=int)

    cfg = ModelConfig(dim=args.dim, heads=args.heads, stages=args.stages, dropout=args.dropout)
    torch.manual_seed(args.seed)
    model = FloorGAT(cfg).to(device)
    if args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
    loss_fn = PanopticLoss(stats["weights"], stats["pos_weight"], args.lam, args.focal_gamma).to(device)

    frozen = [p for s in model.stages[:args.freeze_stages] for p in s.parameters()]
    frozen_ids = {id(p) for p in frozen}
    rest = [p for p in model.parameters() if id(p) not in frozen_ids]
    groups = [{"params": rest, "base_lr": args.lr, "frozen_until": 0, "lr": args.lr}]
    if frozen:
        groups.append({"params": frozen, "base_lr": args.lr, "frozen_until": args.freeze_epochs, "lr": args.lr})
    opt = torch.optim.Adam(groups)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, step, best, bad_evals = 0, 0, -1.0, 0
    last_path = out / "last.pt"
    if last_path.exists() and not args.no_resume:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step, best, bad_evals = ck["epoch"] + 1, ck["step"], ck["best"], ck["bad_evals"]
        if rank == 0:
            print(f"resumed from epoch {ck['epoch']}")
    train_model = DDP(model, device_ids=[local] if device.type == "cuda" else None) if world > 1 else model

    if rank == 0:
        run_cfg = {"args": vars(args), "model": cfg.to_dict(), "params": count_parameters(model), "world": world,
                   "device": torch.cuda.get_device_name(local) if device.type == "cuda" else "cpu",
                   "torch": torch.__version__, "train_tiles": len(pool), "indian_tiles": n_ind,
                   "val_tiles": {k: len(v) for k, v in val_sets.items()},
                   "class_weights": stats["weights"].tolist(), "edge_pos_weight": stats["pos_weight"],
                   "class_counts": stats["counts"].tolist()}
        (out / "run_config.json").write_text(json.dumps(run_cfg, indent=1))
        print(json.dumps({k: v for k, v in run_cfg.items() if k != "args"}))
    wb = None
    if rank == 0 and args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb, name=out.name, config=vars(args), resume="allow")

    stop = False
    for epoch in range(start_epoch, args.epochs):
        rng = np.random.default_rng([args.seed, epoch])
        if finetune and len(ind_idx) and len(fp_idx):
            n_fp = int(round(len(ind_idx) * args.fp_mix / (1 - args.fp_mix)))
            idx = np.concatenate([ind_idx, rng.choice(fp_idx, min(n_fp, len(fp_idx)), replace=False)])
        else:
            idx = np.arange(len(pool))
        batches = pad_shard(make_batches(sizes, idx, args.node_budget, args.max_tiles, rng), rank, world)
        loader = make_loader(pool, batches, args, not args.no_aug, epoch)
        for g in opt.param_groups:
            g["lr"] = lr_for(g, epoch, step, args)
        model.train()
        t0 = time.time()
        acc = np.zeros(8)
        gn_max = 0.0
        for batch in loader:
            for g in opt.param_groups:
                g["lr"] = lr_for(g, epoch, step, args)
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out_t = train_model(batch["x"], batch["edge_index"], batch["edge_attr"])
            total, parts = loss_fn(out_t, batch["y"], batch["edge_y"], batch["edge_mask"], batch["node_src"])
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
            finite = np.isfinite(gn)
            if amp:
                scaler.step(opt)
                scaler.update()
            elif finite:
                opt.step()
            step += 1
            tl, nl, el = float(parts["total"]), float(parts["node"]), float(parts["edge"])
            acc += [tl if np.isfinite(tl) else 0.0, nl if np.isfinite(nl) else 0.0, el if np.isfinite(el) else 0.0, 1,
                    0 if finite else 1, gn if finite else 0.0, batch["x"].shape[0], len(batch["node_ptr"]) - 1]
            gn_max = max(gn_max, gn if finite else 0.0)
            if rank == 0 and step % args.log_every == 0:
                append_jsonl(out / "steps.jsonl", {"step": step, "epoch": epoch, "loss": tl, "node": nl, "edge": el,
                                                   "grad_norm": gn, "lr": opt.param_groups[0]["lr"],
                                                   "scale": float(scaler.get_scale()) if amp else 1.0})
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        tot = reduce_sum(acc.tolist(), device)
        seconds = time.time() - t0
        n_steps = max(tot[3], 1)
        row = {"event": "epoch", "epoch": epoch, "step": step, "loss": tot[0] / n_steps, "loss_node": tot[1] / n_steps,
               "loss_edge": tot[2] / n_steps, "steps": int(tot[3]), "skipped": int(tot[4]),
               "grad_mean": tot[5] / max(tot[3] - tot[4], 1), "grad_max": reduce_max(gn_max, device),
               "nodes": int(tot[6]), "tiles": int(tot[7]), "lr": opt.param_groups[0]["lr"], "seconds": seconds,
               "tiles_per_s": tot[7] / max(seconds, 1e-9),
               "gpu_peak_gb": torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0.0}
        if rank == 0:
            append_jsonl(out / "metrics.jsonl", row)
            print(f"epoch {epoch} loss {row['loss']:.4f} node {row['loss_node']:.4f} edge {row['loss_edge']:.4f} "
                  f"gn {row['grad_mean']:.2f} {seconds:.0f}s")
            if wb:
                wb.log({f"train/{k}": v for k, v in row.items() if isinstance(v, (int, float))}, step=epoch)
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs or stop:
            for name, tiles in val_sets.items():
                s = run_validation(model, tiles, loss_fn, args, rank, world, device)
                if rank == 0:
                    append_jsonl(out / "metrics.jsonl", {"event": "val", "set": name, "epoch": epoch, "summary": s})
                    print(f"  val[{name}] wF1 {s['wf1']:.4f} F1 {s['f1_macro']:.4f} PQ {s['pq']:.4f} "
                          f"edgeF1 {s['edge_f1']:.4f}")
                    if wb:
                        wb.log({f"val_{name}/{k}": v for k, v in s.items() if isinstance(v, float)}, step=epoch)
                if name == args.val_set:
                    score = select_score(s)
                    if score > best:
                        best, bad_evals = score, 0
                        if rank == 0:
                            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "epoch": epoch,
                                        "score": score, "thr": args.thr, "summary": s}, out / "best.pt")
                    else:
                        bad_evals += 1
            if bad_evals >= args.patience:
                stop = True
        if rank == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch, "step": step, "best": best, "bad_evals": bad_evals,
                        "cfg": cfg.to_dict()}, last_path)
        if world > 1:
            dist.barrier()
        if stop:
            if rank == 0:
                print("stopping", "(early stop)" if bad_evals >= args.patience else "(max steps)")
            break
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
