import os
import io
import re
import json
import math
import time
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

cv2.setNumThreads(1)

RASTER_CLASSES = ["background", "wall", "curtain_wall", "railing", "door_single", "door_double", "door_sliding",
                  "door_other", "window", "stairs", "elevator"]
TARGET_TO_RASTER = {"wall": 1, "curtain_wall": 2, "railing": 3, "door_single": 4, "door_double": 5,
                    "door_sliding": 6, "door_other": 7, "window": 8, "stairs": 9, "elevator": 10}
WALL_LIKE = {1, 2, 3}
DOOR_IDS = {4, 5, 6, 7}
OPENING_IDS = {4, 5, 6, 7, 8}
THING_IDS = {4, 5, 6, 7, 8, 9, 10}
FILLED_IDS = {9, 10}
PALETTE = np.array([[255, 255, 255], [31, 78, 156], [23, 190, 207], [140, 86, 75], [214, 39, 40], [255, 127, 14],
                    [227, 119, 194], [188, 189, 34], [44, 160, 44], [148, 103, 189], [127, 127, 127]], dtype=np.uint8)


@dataclass
class RenderConfig:
    mm_per_px: float = 10.0
    margin_px: int = 16
    line_px: tuple = (1, 3)
    mask_stroke_px: int = 5
    wall_stroke_px: int = 1
    wall_close_mm: float = 250.0
    arc_samples: int = 24


def target_lut(targets):
    lut = np.zeros(len(targets), dtype=np.uint8)
    for i, t in enumerate(targets):
        lut[i] = TARGET_TO_RASTER.get(t, 0)
    return lut


def colorize(sem):
    return PALETTE[np.clip(sem, 0, len(PALETTE) - 1)]


def _circumcircle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return np.array([ux, uy]), float(np.hypot(ax - ux, ay - uy))


def _arc_points(p, n):
    cc = _circumcircle(p[0], p[1], p[3])
    if cc is None:
        return p
    c, r = cc
    a0 = math.atan2(p[0][1] - c[1], p[0][0] - c[0])
    a1 = math.atan2(p[3][1] - c[1], p[3][0] - c[0])
    am = math.atan2(p[1][1] - c[1], p[1][0] - c[0])
    span = (a1 - a0) % (2 * math.pi)
    if (am - a0) % (2 * math.pi) > span:
        span = span - 2 * math.pi
    t = a0 + np.linspace(0, 1, n) * span
    return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], 1)


def primitive_polylines(pts, ptype, arc_samples=24):
    out = []
    for p, t in zip(pts.astype(np.float64), ptype):
        if t == 0:
            out.append(p[[0, 3]])
        elif t == 1:
            out.append(_arc_points(p, arc_samples))
        elif t in (2, 3):
            c = p.mean(0)
            rx, ry = abs(p[0][0] - c[0]), abs(p[1][1] - c[1])
            rx = rx if rx > 0 else np.linalg.norm(p[0] - c)
            ry = ry if ry > 0 else rx
            a = np.linspace(0, 2 * math.pi, max(16, arc_samples))
            out.append(np.stack([c[0] + rx * np.cos(a), c[1] + ry * np.sin(a)], 1))
        else:
            out.append(p)
    return out


def _to_px(poly, origin, s):
    return np.round((poly - origin) / s * 16).astype(np.int32)


def line_kernels(length_px):
    k = 2 * int(round(length_px)) + 1
    return [np.ones((1, k), np.uint8), np.ones((k, 1), np.uint8)]


def fill_small_holes(mask, max_area_px):
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask == 0).astype(np.uint8), connectivity=4)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= max_area_px
    out = mask.copy()
    out[small[lab] & (mask == 0)] = 1
    return out


def close_wall_faces(mask, gap_px):
    out = mask.copy()
    for ker in line_kernels(gap_px):
        out |= cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    return fill_small_holes(out, (2 * gap_px) ** 2)


def render_tile(tile, lut, cfg, rng, frame=None):
    s = cfg.mm_per_px
    pts = tile["pts"].astype(np.float64)
    polys = primitive_polylines(pts, tile["ptype"], cfg.arc_samples)
    if frame is None:
        allp = np.concatenate(polys, 0) if polys else np.zeros((1, 2))
        origin = allp.min(0) - cfg.margin_px * s
        size = np.ceil((allp.max(0) - origin) / s).astype(int) + cfg.margin_px
        W, H = int(size[0]), int(size[1])
    else:
        origin, (W, H) = np.zeros(2), frame
    px = [_to_px(q, origin, s) for q in polys]
    img = np.full((H, W), 255, np.uint8)
    lw = int(rng.integers(cfg.line_px[0], cfg.line_px[1] + 1))
    cv2.polylines(img, [q.reshape(-1, 1, 2) for q in px], False, 0, lw, cv2.LINE_AA, shift=4)
    cls = lut[tile["y"].astype(np.int64)]
    inst = tile["inst"].astype(np.int64)
    sem = np.zeros((H, W), np.uint8)
    imap = np.zeros((H, W), np.uint16)
    wall = np.zeros((H, W), np.uint8)
    wall_idx = np.where(cls == 1)[0]
    if len(wall_idx):
        cv2.polylines(wall, [px[i].reshape(-1, 1, 2) for i in wall_idx], False, 1, cfg.wall_stroke_px, cv2.LINE_8, shift=4)
        wall = close_wall_faces(wall, cfg.wall_close_mm / s)
        sem[wall > 0] = 1
    for c in (2, 3):
        idx = np.where(cls == c)[0]
        if len(idx):
            m = np.zeros((H, W), np.uint8)
            cv2.polylines(m, [px[i].reshape(-1, 1, 2) for i in idx], False, 1, cfg.mask_stroke_px, cv2.LINE_8, shift=4)
            sem[m > 0] = c
    next_id = 1
    for c in (9, 10, 8, 7, 6, 5, 4):
        idx = np.where(cls == c)[0]
        if not len(idx):
            continue
        groups = {}
        for i in idx:
            groups.setdefault(int(inst[i]) if inst[i] >= 0 else -1 - int(i), []).append(i)
        for key, members in groups.items():
            m = np.zeros((H, W), np.uint8)
            cv2.polylines(m, [px[i].reshape(-1, 1, 2) for i in members], False, 1, cfg.mask_stroke_px, cv2.LINE_8, shift=4)
            if key >= 0:
                hull_pts = np.concatenate([px[i] for i in members], 0)
                if len(hull_pts) >= 3:
                    hull = cv2.convexHull(hull_pts.reshape(-1, 1, 2))
                    cv2.fillPoly(m, [hull], 1, cv2.LINE_8, shift=4)
                sel = m > 0
                sem[sel] = c
                imap[sel] = next_id
                next_id += 1
            else:
                sem[m > 0] = c
    return img, sem, imap


def render_shard_job(args):
    shard_path, out_dir, split, lut, cfg_dict, seed, max_tiles = args
    import fpgraph
    cfg = RenderConfig(**cfg_dict)
    rng = np.random.default_rng(seed)
    S = fpgraph.load_shard(shard_path)
    n = len(S["node_ptr"]) - 1
    rows = []
    out = Path(out_dir) / split
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(shard_path).stem
    for k in range(min(n, max_tiles) if max_tiles else n):
        t = fpgraph.get_tile(S, k)
        if len(t["y"]) < 8:
            continue
        img, sem, imap = render_tile(t, lut, cfg, rng)
        tid = f"{stem}_{k:04d}"
        cv2.imwrite(str(out / f"{tid}_img.png"), img)
        cv2.imwrite(str(out / f"{tid}_sem.png"), sem)
        cv2.imwrite(str(out / f"{tid}_inst.png"), imap)
        counts = np.bincount(sem.ravel(), minlength=len(RASTER_CLASSES))
        rows.append({"id": tid, "split": split, "h": img.shape[0], "w": img.shape[1], "n_inst": int(imap.max()),
                     **{f"px_{c}": int(v) for c, v in zip(RASTER_CLASSES, counts)}})
    return rows


def render_split(shards, out_dir, split, lut, cfg, workers=4, max_tiles_per_shard=None, seed=0):
    from multiprocessing import Pool
    jobs = [(str(p), str(out_dir), split, lut, asdict(cfg), seed + i, max_tiles_per_shard) for i, p in enumerate(shards)]
    t0 = time.time()
    if workers <= 1:
        results = [render_shard_job(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            results = list(pool.imap_unordered(render_shard_job, jobs))
    df = pd.DataFrame([r for rows in results for r in rows]).sort_values("id").reset_index(drop=True)
    print(f"{split}: {len(df)} tiles rendered from {len(shards)} shards in {time.time() - t0:.0f}s")
    return df


def class_weights(df, power=0.5, cap=10.0):
    px = np.array([df[f"px_{c}"].sum() for c in RASTER_CLASSES], dtype=np.float64)
    freq = px / max(px.sum(), 1)
    present = px > 0
    w = np.where(present, (np.median(freq[present]) / np.maximum(freq, 1e-12)) ** power, 0.0)
    return np.clip(w, 0.05, cap) * present, px


TEXT_WORDS = ["M.BEDROOM", "BEDROOM", "TOILET", "KITCHEN", "LIVING", "DINING", "FOYER", "BALCONY", "LOBBY", "CORRIDOR",
              "UTILITY", "SHAFT", "OT", "ICU", "WARD", "NURSE STN", "STORE", "LIFT", "STAIRCASE", "UP", "DN", "D1", "W2",
              "3650X4170", "1580X2480", "1650 WIDE", "+101.15", "SCALE 1:100", "4500", "2300", "900", "1200"]


def add_random_text(img, rng, mm_per_px=10.0, n_range=(3, 18)):
    h, w = img.shape
    for _ in range(int(rng.integers(*n_range))):
        word = TEXT_WORDS[int(rng.integers(len(TEXT_WORDS)))]
        height_px = rng.uniform(120, 420) / mm_per_px
        scale = height_px / 22.0
        thick = max(1, int(round(scale * rng.uniform(1.0, 2.2))))
        x, y = int(rng.integers(0, max(1, w - 20))), int(rng.integers(int(height_px) + 1, max(int(height_px) + 2, h)))
        font = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX][int(rng.integers(3))]
        cv2.putText(img, word, (x, y), font, scale, 0, thick, cv2.LINE_AA)
    return img


def scan_degrade(img, rng, severity=None):
    sev = rng.uniform(0.2, 1.0) if severity is None else severity
    out = img.astype(np.float32)
    if rng.random() < 0.6 * sev + 0.2:
        k = 2 * int(rng.integers(0, 1 + int(2 * sev))) + 1
        if k > 1:
            out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.5:
        out = 255 - (255 - out) * rng.uniform(0.55, 1.0)
    out = out + rng.normal(0, 12 * sev, out.shape)
    if rng.random() < 0.4 * sev:
        drop = rng.random(out.shape) < 0.04 * sev
        out[drop & (out < 128)] = 255
    if rng.random() < 0.3 * sev:
        sp = rng.random(out.shape)
        out[sp < 0.002 * sev] = 0
        out[sp > 1 - 0.002 * sev] = 255
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.5 * sev + 0.1:
        q = int(rng.integers(int(90 - 55 * sev), 96))
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return out


def degrade_fixed(img, level, seed=0):
    if level == "clean":
        return img
    rng = np.random.default_rng(seed)
    return scan_degrade(img, rng, {"mild": 0.4, "heavy": 1.0}[level])


class TileDataset:
    def __init__(self, root, ids, crop=512, train=True, scale_range=(0.8, 1.25), p_scan=0.5, p_text=0.7,
                 p_invert=0.15, mm_per_px=10.0, seed=0, fixed_degrade=None):
        self.root, self.ids, self.crop, self.train = Path(root), list(ids), crop, train
        self.scale_range, self.p_scan, self.p_text, self.p_invert = scale_range, p_scan, p_text, p_invert
        self.mm_per_px, self.seed, self.fixed_degrade = mm_per_px, seed, fixed_degrade

    def __len__(self):
        return len(self.ids)

    def _load(self, tid):
        img = cv2.imread(str(self.root / f"{tid}_img.png"), cv2.IMREAD_GRAYSCALE)
        sem = cv2.imread(str(self.root / f"{tid}_sem.png"), cv2.IMREAD_UNCHANGED)
        return img, sem

    def _pad(self, img, sem, size):
        h, w = img.shape
        ph, pw = max(0, size - h), max(0, size - w)
        if ph or pw:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=255)
            sem = cv2.copyMakeBorder(sem, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        return img, sem

    def __getitem__(self, i):
        import torch
        tid = self.ids[i]
        img, sem = self._load(tid)
        if self.train:
            rng = np.random.default_rng((self.seed * 1_000_003 + i * 7919 + int(time.time() * 1e6)) % (2 ** 32))
            f = rng.uniform(*self.scale_range)
            if abs(f - 1) > 0.02:
                img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
                sem = cv2.resize(sem, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            img, sem = self._pad(img, sem, self.crop)
            h, w = img.shape
            fg = np.argwhere(sem[:: 8, :: 8] > 0)
            if len(fg) and rng.random() < 0.8:
                cy, cx = fg[int(rng.integers(len(fg)))] * 8
                y0 = int(np.clip(cy - self.crop // 2 + rng.integers(-self.crop // 4, self.crop // 4 + 1), 0, h - self.crop))
                x0 = int(np.clip(cx - self.crop // 2 + rng.integers(-self.crop // 4, self.crop // 4 + 1), 0, w - self.crop))
            else:
                y0, x0 = int(rng.integers(0, h - self.crop + 1)), int(rng.integers(0, w - self.crop + 1))
            img = np.ascontiguousarray(img[y0:y0 + self.crop, x0:x0 + self.crop])
            sem = np.ascontiguousarray(sem[y0:y0 + self.crop, x0:x0 + self.crop])
            k = int(rng.integers(4))
            img, sem = np.rot90(img, k), np.rot90(sem, k)
            if rng.random() < 0.5:
                img, sem = img[:, ::-1], sem[:, ::-1]
            img, sem = np.ascontiguousarray(img), np.ascontiguousarray(sem)
            if rng.random() < self.p_text:
                img = add_random_text(img, rng, self.mm_per_px)
            if rng.random() < self.p_scan:
                img = scan_degrade(img, rng)
            if rng.random() < self.p_invert:
                img = 255 - img
        else:
            img, sem = self._pad(img, sem, self.crop)
            img, sem = img[: self.crop, : self.crop], sem[: self.crop, : self.crop]
            if self.fixed_degrade:
                img = degrade_fixed(img, self.fixed_degrade, seed=i)
        x = torch.from_numpy((255 - img.astype(np.float32)) / 255.0)[None]
        y = torch.from_numpy(sem.astype(np.int64))
        return x, y


def make_model(encoder="resnet34", weights="imagenet", n_classes=len(RASTER_CLASSES)):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=n_classes)


class SegLoss:
    def __init__(self, weights, dice_weight=0.5, device="cpu"):
        import torch
        import segmentation_models_pytorch as smp
        self.ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        self.dice = smp.losses.DiceLoss(mode="multiclass", classes=list(range(1, len(RASTER_CLASSES))))
        self.dw = dice_weight

    def __call__(self, logits, y):
        return self.ce(logits.float(), y) + self.dw * self.dice(logits.float(), y)


def confusion(pred, gt, n=len(RASTER_CLASSES)):
    return np.bincount(gt.ravel().astype(np.int64) * n + pred.ravel().astype(np.int64), minlength=n * n).reshape(n, n)


def iou_from_conf(conf):
    tp = np.diag(conf).astype(np.float64)
    denom = conf.sum(0) + conf.sum(1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    return iou


def evaluate(model, loader, device, amp=True, max_batches=None):
    import torch
    model.eval()
    conf = np.zeros((len(RASTER_CLASSES),) * 2, dtype=np.int64)
    with torch.no_grad():
        for b, (x, y) in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.startswith("cuda")):
                logits = model(x.to(device, non_blocking=True))
            conf += confusion(logits.argmax(1).cpu().numpy(), y.numpy())
    iou = iou_from_conf(conf)
    return {"miou_fg": float(np.nanmean(iou[1:])), "iou": dict(zip(RASTER_CLASSES, np.round(np.nan_to_num(iou, nan=-1), 4).tolist())),
            "key_miou": float(np.nanmean(iou[[1, 4, 5, 6, 8]]))}


def lr_at(frac, base, warmup=0.03, floor=0.02):
    if frac < warmup:
        return base * max(frac / warmup, 0.01)
    p = min(1.0, (frac - warmup) / (1 - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def train_timeboxed(model, train_loader, val_loader, loss_fn, minutes, ckpt_dir, base_lr=3e-4, weight_decay=1e-4,
                    device="cuda", amp=True, val_every_min=8.0, val_batches=None, log=print, resume=None):
    import torch
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    net = model
    if device.startswith("cuda") and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(model)
    net.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.startswith("cuda"))
    history, best, step, epoch, spent = [], -1.0, 0, 0, 0.0
    if resume and Path(resume).exists():
        st = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        history, best, step, epoch, spent = st["history"], st["best"], st["step"], st["epoch"], st["spent"]
        log(f"resumed at epoch {epoch}, {spent / 60:.1f} min already trained, best {best:.4f}")
    budget = minutes * 60.0
    if spent >= budget:
        log(f"budget of {minutes} min already used; raise TRAIN_MINUTES to continue")
        return history
    t_start = time.time() - spent
    next_val = (spent // (val_every_min * 60) + 1) * val_every_min * 60
    done = False
    while not done:
        net.train()
        run, nb = 0.0, 0
        for x, y in train_loader:
            elapsed = time.time() - t_start
            if elapsed >= budget:
                done = True
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(elapsed / budget, base_lr)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.startswith("cuda")):
                logits = net(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            run += float(loss.detach())
            nb += 1
            step += 1
            if time.time() - t_start >= next_val:
                break
        epoch += 1
        elapsed = time.time() - t_start
        rec = {"epoch": epoch, "step": step, "minutes": round(elapsed / 60, 2), "train_loss": round(run / max(nb, 1), 4),
               "lr": opt.param_groups[0]["lr"]}
        if nb == 0 and done and history and "val_key_miou" in history[-1]:
            break
        if elapsed >= next_val or done:
            ev = evaluate(net, val_loader, device, amp, val_batches)
            rec.update({"val_miou_fg": round(ev["miou_fg"], 4), "val_key_miou": round(ev["key_miou"], 4)})
            next_val += val_every_min * 60
            score = ev["key_miou"] if np.isfinite(ev["key_miou"]) else 0.0
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "classes": RASTER_CLASSES, "val": ev}, ckpt_dir / "best.pt")
                rec["saved"] = "best"
        history.append(rec)
        log(rec)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "history": history,
                    "best": best, "step": step, "epoch": epoch, "spent": time.time() - t_start}, ckpt_dir / "last.pt")
    return history


def predict_image(model, img, device, tile=1024, overlap=128, amp=True, batch=4):
    import torch
    model.eval()
    h, w = img.shape
    H, W = max(h, tile), max(w, tile)
    pad = np.full((H, W), 255, np.uint8)
    pad[:h, :w] = img
    stride = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if ys[-1] != H - tile:
        ys.append(H - tile)
    if xs[-1] != W - tile:
        xs.append(W - tile)
    n = len(RASTER_CLASSES)
    acc = np.zeros((n, H, W), np.float16)
    cnt = np.zeros((H, W), np.float16)
    win = np.outer(np.hanning(tile) * 0.9 + 0.1, np.hanning(tile) * 0.9 + 0.1).astype(np.float16)
    coords = [(y, x) for y in ys for x in xs]
    with torch.no_grad():
        for i in range(0, len(coords), batch):
            chunk = coords[i:i + batch]
            xb = np.stack([(255 - pad[y:y + tile, x:x + tile].astype(np.float32)) / 255.0 for y, x in chunk])[:, None]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.startswith("cuda")):
                prob = torch.softmax(model(torch.from_numpy(xb).to(device)).float(), 1).cpu().numpy().astype(np.float16)
            for (y, x), p in zip(chunk, prob):
                acc[:, y:y + tile, x:x + tile] += p * win
                cnt[y:y + tile, x:x + tile] += win
    pred = np.zeros((h, w), np.uint8)
    conf = np.zeros((h, w), np.float32)
    for y0 in range(0, h, 256):
        y1 = min(y0 + 256, h)
        band = acc[:, y0:y1, :w] / np.maximum(cnt[y0:y1, :w], 1e-3)
        pred[y0:y1] = band.argmax(0)
        conf[y0:y1] = band.max(0)
    return pred, conf


def instance_pq(pred, gt_sem, gt_inst, classes=sorted(THING_IDS), min_px=30):
    out = {}
    for c in classes:
        g = np.where((gt_sem == c) & (gt_inst > 0), gt_inst, 0).astype(np.int64)
        gids = np.unique(g[g > 0])
        n, plab, st, _ = cv2.connectedComponentsWithStats((pred == c).astype(np.uint8), connectivity=8)
        keep = np.where(st[:, cv2.CC_STAT_AREA] >= min_px)[0]
        keep = keep[keep > 0]
        if not len(gids) and not len(keep):
            continue
        gmap = np.zeros(int(g.max()) + 1, np.int64)
        gmap[gids] = np.arange(1, len(gids) + 1)
        gi = gmap[g]
        pair = np.bincount(gi.ravel() * n + plab.ravel(), minlength=(len(gids) + 1) * n).reshape(len(gids) + 1, n)
        inter = pair[1:][:, keep].astype(np.float64)
        area_g = pair[1:].sum(1).astype(np.float64)
        area_p = st[keep, cv2.CC_STAT_AREA].astype(np.float64)
        ious = inter / np.maximum(area_g[:, None] + area_p[None, :] - inter, 1)
        matched = ious > 0.5
        tp = int(matched.any(1).sum())
        out[RASTER_CLASSES[c]] = {"tp": tp, "fp": len(keep) - int(matched.any(0).sum()), "fn": len(gids) - tp,
                                  "iou_sum": float(ious[matched].sum())}
    return out


def summarize_pq(per_tile):
    agg = {}
    for d in per_tile:
        for c, v in d.items():
            a = agg.setdefault(c, {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0})
            for k in a:
                a[k] += v[k]
    rows = []
    for c, a in agg.items():
        denom = a["tp"] + 0.5 * a["fp"] + 0.5 * a["fn"]
        sq = a["iou_sum"] / a["tp"] if a["tp"] else 0.0
        rq = a["tp"] / denom if denom else 0.0
        rows.append({"class": c, "PQ": round(sq * rq, 4), "SQ": round(sq, 4), "RQ": round(rq, 4), **{k: a[k] for k in ("tp", "fp", "fn")}})
    return pd.DataFrame(rows)


def load_tile(root, tid, crop=1024):
    img = cv2.imread(str(Path(root) / f"{tid}_img.png"), cv2.IMREAD_GRAYSCALE)
    sem = cv2.imread(str(Path(root) / f"{tid}_sem.png"), cv2.IMREAD_UNCHANGED)
    inst = cv2.imread(str(Path(root) / f"{tid}_inst.png"), cv2.IMREAD_UNCHANGED)
    out = []
    for a, v in ((img, 255), (sem, 0), (inst, 0)):
        h, w = a.shape
        a = cv2.copyMakeBorder(a, 0, max(0, crop - h), 0, max(0, crop - w), cv2.BORDER_CONSTANT, value=v)
        out.append(a[:crop, :crop])
    return out


def evaluate_split(model, root, ids, device, level="clean", crop=1024, batch=8, amp=True, log_every=0):
    import torch
    model.eval()
    conf = np.zeros((len(RASTER_CLASSES),) * 2, dtype=np.int64)
    pq = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            tiles = [load_tile(root, t, crop) for t in chunk]
            imgs = [degrade_fixed(img, level, seed=i + j) for j, (img, _, _) in enumerate(tiles)]
            xb = torch.from_numpy(np.stack([(255 - im.astype(np.float32)) / 255.0 for im in imgs])[:, None]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.startswith("cuda")):
                pred = model(xb).argmax(1).cpu().numpy().astype(np.uint8)
            for p, (_, sem, inst) in zip(pred, tiles):
                conf += confusion(p, sem)
                pq.append(instance_pq(p, sem, inst))
            if log_every and (i // batch) % log_every == 0:
                print(f"  {level}: {i + len(chunk)}/{len(ids)} tiles, {time.time() - t0:.0f}s")
    iou = iou_from_conf(conf)
    return {"level": level, "miou_fg": float(np.nanmean(iou[1:])), "key_miou": float(np.nanmean(iou[[1, 4, 5, 6, 8]])),
            "iou": dict(zip(RASTER_CLASSES, np.round(np.nan_to_num(iou, nan=-1), 4).tolist())), "pq": summarize_pq(pq)}


def _iter_entities(entities, depth=0):
    for e in entities:
        t = e.dxftype()
        if t in ("INSERT", "DIMENSION", "LEADER", "MLEADER", "MULTILEADER", "ACAD_TABLE") and depth < 10:
            try:
                yield from _iter_entities(e.virtual_entities(), depth + 1)
            except Exception:
                continue
            if t == "INSERT":
                for a in getattr(e, "attribs", []):
                    yield a
        else:
            yield e


def manifest_region(manifest, margin_mm=3000.0):
    xs, ys = [], []
    for r in manifest["rooms"]:
        p = np.array(r["polygon"])
        xs += [p[:, 0].min(), p[:, 0].max()]
        ys += [p[:, 1].min(), p[:, 1].max()]
    return (min(xs) - margin_mm, min(ys) - margin_mm, max(xs) + margin_mm, max(ys) + margin_mm)


def render_dxf(path, region, mm_per_px=10.0, line_px=2, with_text=True):
    import ezdxf
    from ezdxf import path as ezpath
    doc = ezdxf.readfile(str(path))
    hidden = {l.dxf.name for l in doc.layers if l.is_off() or l.is_frozen()}
    x0, y0, x1, y1 = region
    W, H = int(math.ceil((x1 - x0) / mm_per_px)), int(math.ceil((y1 - y0) / mm_per_px))
    img = np.full((H, W), 255, np.uint8)
    polys, n_text = [], 0
    for e in _iter_entities(doc.modelspace()):
        if e.dxf.get("layer", "0") in hidden:
            continue
        t = e.dxftype()
        if t in ("TEXT", "MTEXT", "ATTRIB"):
            if not with_text:
                continue
            try:
                txt = e.plain_text() if t == "MTEXT" else e.dxf.text
                ins = e.dxf.insert
                h = float((e.dxf.get("char_height", 250.0) if t == "MTEXT" else e.dxf.get("height", 250.0)) or 250.0)
            except Exception:
                continue
            if not txt or not (x0 <= ins.x <= x1 and y0 <= ins.y <= y1):
                continue
            px, py = int((ins.x - x0) / mm_per_px), int((y1 - ins.y) / mm_per_px)
            scale = max(h / mm_per_px / 22.0, 0.2)
            cv2.putText(img, txt.replace("\n", " ")[:60], (px, py), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, max(1, int(scale)), cv2.LINE_AA)
            n_text += 1
            continue
        if t not in ("LINE", "ARC", "CIRCLE", "LWPOLYLINE", "POLYLINE", "ELLIPSE", "SPLINE"):
            continue
        try:
            pts = np.array([(v.x, v.y) for v in ezpath.make_path(e).flattening(mm_per_px / 2)])
        except Exception:
            continue
        if len(pts) < 2 or pts[:, 0].max() < x0 or pts[:, 0].min() > x1 or pts[:, 1].max() < y0 or pts[:, 1].min() > y1:
            continue
        q = np.stack([(pts[:, 0] - x0) / mm_per_px, (y1 - pts[:, 1]) / mm_per_px], 1)
        polys.append(np.round(q * 16).astype(np.int32).reshape(-1, 1, 2))
    cv2.polylines(img, polys, False, 0, line_px, cv2.LINE_AA, shift=4)
    return img, {"region_mm": list(region), "mm_per_px": mm_per_px, "size_px": [W, H], "n_polylines": len(polys), "n_text": n_text}


def px_to_mm(xy, region, s):
    return np.stack([region[0] + xy[:, 0] * s, region[3] - xy[:, 1] * s], 1)


def mm_to_px(xy, region, s):
    return np.stack([(xy[:, 0] - region[0]) / s, (region[3] - xy[:, 1]) / s], 1)


def seal_walls(pred, s, door_gap_mm=2600.0, small_gap_mm=40.0, near_mm=150.0):
    boundary = np.isin(pred, list(WALL_LIKE | {6, 8})).astype(np.uint8)
    k = max(3, 2 * int(round(small_gap_mm / s / 2)) + 1)
    boundary = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    swing = np.isin(pred, [4, 5, 7]).astype(np.uint8)
    L = 2 * int(round(door_gap_mm / s / 2)) + 1
    ch = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    cv_ = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    r = 2 * int(round(near_mm / s / 2)) + 1
    near = cv2.dilate(swing, np.ones((r, r), np.uint8))
    plug = ((ch | cv_) & near & (1 - boundary)).astype(np.uint8)
    return (boundary | plug).astype(np.uint8), plug


def open_space_pockets(sealed, exterior, s, close_mm=2000.0):
    building = (1 - exterior.astype(np.uint8)).astype(np.uint8)
    L = 2 * int(round(close_mm / s)) + 1
    closed = cv2.morphologyEx(building, cv2.MORPH_CLOSE, np.ones((1, L), np.uint8))
    closed &= cv2.morphologyEx(building, cv2.MORPH_CLOSE, np.ones((L, 1), np.uint8))
    return (closed & exterior.astype(np.uint8) & (1 - sealed)).astype(np.uint8)


def reclaim_plugs(room_lab, rooms, plug, s, max_mm=1200.0):
    from scipy import ndimage
    free = room_lab == 0
    dist, (iy, ix) = ndimage.distance_transform_edt(free, return_indices=True)
    take = (plug > 0) & free & (dist <= max_mm / s)
    room_lab[take] = room_lab[iy[take], ix[take]]
    counts = np.bincount(room_lab.ravel(), minlength=len(rooms) + 1)
    for r in rooms:
        r["area_m2"] = round(float(counts[r["rid"]]) * s * s / 1e6, 3)
    return room_lab


def extract_rooms(sealed, s, min_area_m2=0.5, open_close_mm=2000.0, plug=None):
    free = (1 - sealed).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(free, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist()) - {0}
    exterior = np.isin(lab, list(border)) & (free > 0)
    min_px = min_area_m2 * 1e6 / (s * s)
    rooms, room_lab = [], np.zeros(lab.shape, dtype=np.int32)

    def add(mask_sub, x, y, area, cx, cy, enclosure):
        rid = len(rooms) + 1
        room_lab[y:y + mask_sub.shape[0], x:x + mask_sub.shape[1]][mask_sub] = rid
        rooms.append({"rid": rid, "area_m2": round(float(area) * s * s / 1e6, 3), "enclosure": enclosure,
                      "bbox_px": [int(x), int(y), int(mask_sub.shape[1]), int(mask_sub.shape[0])], "centroid_px": [float(cx), float(cy)]})

    for k in range(1, n):
        if k in border or stats[k, cv2.CC_STAT_AREA] < min_px:
            continue
        x, y, bw, bh = stats[k, :4]
        add(lab[y:y + bh, x:x + bw] == k, x, y, stats[k, cv2.CC_STAT_AREA], cent[k][0], cent[k][1], "closed")
    if plug is not None:
        room_lab = reclaim_plugs(room_lab, rooms, plug, s)
    if open_close_mm:
        pockets = open_space_pockets(sealed, exterior, s, open_close_mm)
        m, plab, pst, pc = cv2.connectedComponentsWithStats(pockets, connectivity=4)
        for k in range(1, m):
            if pst[k, cv2.CC_STAT_AREA] < max(min_px, 2e6 / (s * s)):
                continue
            x, y, bw, bh = pst[k, :4]
            add(plab[y:y + bh, x:x + bw] == k, x, y, pst[k, cv2.CC_STAT_AREA], pc[k][0], pc[k][1], "open")
        exterior = exterior & (room_lab == 0)
    return rooms, room_lab, exterior


def room_polygons(room_lab, rooms, region, s, eps_mm=20.0):
    from shapely.geometry import Polygon
    for r in rooms:
        x, y, bw, bh = r["bbox_px"]
        m = (room_lab[y:y + bh, x:x + bw] == r["rid"]).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, eps_mm / s, True).reshape(-1, 2).astype(np.float64) + [x, y]
        r["polygon"] = np.round(px_to_mm(c, region, s), 1).tolist()
    return rooms


def extract_openings(pred, conf, room_lab, exterior, region, s, min_area_mm2=150000.0):
    out = []
    mask = np.isin(pred, list(OPENING_IDS)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_px = min_area_mm2 / (s * s)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_px:
            continue
        x, y, bw, bh = stats[k, :4]
        sub = lab[y:y + bh, x:x + bw] == k
        cls_vals = pred[y:y + bh, x:x + bw][sub]
        c = int(np.bincount(cls_vals, minlength=len(RASTER_CLASSES)).argmax())
        pts = (np.argwhere(sub)[:, ::-1] + [x, y]).astype(np.float32)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
        width_mm = float(max(rw, rh) * s)
        pad = 20
        ya, yb, xa, xb = max(0, y - pad), min(pred.shape[0], y + bh + pad), max(0, x - pad), min(pred.shape[1], x + bw + pad)
        grown = cv2.dilate((lab[ya:yb, xa:xb] == k).astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) > 0
        touch = np.bincount(room_lab[ya:yb, xa:xb][grown], minlength=1)
        touch[0] = 0
        order = [int(i) for i in np.argsort(-touch) if touch[i] > 0][:2]
        connects = [f"P{i:03d}" for i in order]
        if len(connects) < 2 and exterior[ya:yb, xa:xb][grown].any():
            connects.append("EXTERIOR")
        out.append({"kind": "door" if c in DOOR_IDS else "window", "class": RASTER_CLASSES[c],
                    "position": np.round(px_to_mm(np.array([[cx, cy]]), region, s)[0], 1).tolist(),
                    "width_mm": round(width_mm, 1), "confidence": round(float(conf[y:y + bh, x:x + bw][sub].mean()), 3),
                    "connects": connects})
    return out


def room_adjacency(room_lab, rooms, s, gap_mm=350.0, min_shared_mm=300.0):
    d = 2 * int(round(gap_mm / s)) + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    edges = {}
    bnd = room_lab > 0
    diff = np.zeros_like(bnd)
    diff[1:] |= room_lab[1:] != room_lab[:-1]
    diff[:-1] |= room_lab[:-1] != room_lab[1:]
    diff[:, 1:] |= room_lab[:, 1:] != room_lab[:, :-1]
    diff[:, :-1] |= room_lab[:, :-1] != room_lab[:, 1:]
    bnd &= diff
    for r in rooms:
        x, y, bw, bh = r["bbox_px"]
        pad = d
        ya, yb, xa, xb = max(0, y - pad), min(room_lab.shape[0], y + bh + pad), max(0, x - pad), min(room_lab.shape[1], x + bw + pad)
        sub = room_lab[ya:yb, xa:xb]
        own = (sub == r["rid"]).astype(np.uint8)
        ring = cv2.dilate(own, ker) & (1 - own)
        others = sub[(ring > 0) & bnd[ya:yb, xa:xb] & (sub != r["rid"])]
        if not len(others):
            continue
        for o, cnt in zip(*np.unique(others, return_counts=True)):
            boundary_len_mm = cnt * s
            key = tuple(sorted((r["rid"], int(o))))
            edges[key] = max(edges.get(key, 0.0), boundary_len_mm)
    return [{"a": f"P{a:03d}", "b": f"P{b:03d}", "shared_wall_mm": round(v, 0)} for (a, b), v in edges.items() if v >= min_shared_mm]


def rasterize_manifest_rooms(manifest, region, s, shape, statuses=("room", "unlabelled")):
    lab = np.zeros(shape, np.int32)
    ids = []
    for r in manifest["rooms"]:
        if r.get("status", "room") not in statuses:
            continue
        ids.append(r["id"])
        poly = mm_to_px(np.array(r["polygon"], dtype=np.float64), region, s)
        cv2.fillPoly(lab, [np.round(poly * 16).astype(np.int32).reshape(-1, 1, 2)], len(ids), cv2.LINE_8, shift=4)
    return lab, ids


def match_rooms(pred_lab, n_pred, gt_lab, n_gt, iou_thr=0.5):
    from scipy.optimize import linear_sum_assignment
    pair = np.bincount(gt_lab.ravel().astype(np.int64) * (n_pred + 1) + pred_lab.ravel().astype(np.int64),
                       minlength=(n_gt + 1) * (n_pred + 1)).reshape(n_gt + 1, n_pred + 1)
    inter = pair[1:, 1:].astype(np.float64)
    area_g = pair[1:, :].sum(1).astype(np.float64)
    area_p = pair[:, 1:].sum(0).astype(np.float64)
    iou = inter / np.maximum(area_g[:, None] + area_p[None, :] - inter, 1)
    gi, pi = linear_sum_assignment(-iou)
    keep = iou[gi, pi] >= iou_thr
    return list(zip(gi[keep] + 1, pi[keep] + 1)), iou, area_g, area_p


def match_points(pred_xy, gt_xy, max_dist):
    from scipy.optimize import linear_sum_assignment
    if not len(pred_xy) or not len(gt_xy):
        return []
    d = np.linalg.norm(np.asarray(gt_xy)[:, None, :] - np.asarray(pred_xy)[None, :, :], axis=2)
    gi, pi = linear_sum_assignment(d)
    return [(int(g), int(p)) for g, p in zip(gi, pi) if d[g, p] <= max_dist]


def edge_f1(pred_edges, gt_edges):
    tp = len(pred_edges & gt_edges)
    p = tp / len(pred_edges) if pred_edges else 0.0
    r = tp / len(gt_edges) if gt_edges else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "pred_edges": len(pred_edges), "gt_edges": len(gt_edges), "tp": tp}


def compare_with_track_a(manifest, rooms, room_lab, openings, adjacency, region, s):
    gt_lab, gt_ids = rasterize_manifest_rooms(manifest, region, s, room_lab.shape)
    pairs, iou, area_g, area_p = match_rooms(room_lab, len(rooms), gt_lab, len(gt_ids))
    g2p = {gt_ids[g - 1]: f"P{p:03d}" for g, p in pairs}
    p2g = {v: k for k, v in g2p.items()}
    area_err = [abs(area_p[p - 1] - area_g[g - 1]) / area_g[g - 1] * 100 for g, p in pairs]
    room_metrics = {
        "gt_rooms": len(gt_ids), "pred_rooms": len(rooms), "matched_iou50": len(pairs),
        "recall": round(len(pairs) / max(len(gt_ids), 1), 4), "precision": round(len(pairs) / max(len(rooms), 1), 4),
        "mean_iou_matched": round(float(np.mean([iou[g - 1, p - 1] for g, p in pairs])) if pairs else 0.0, 4),
        "median_area_error_pct": round(float(np.median(area_err)) if area_err else 0.0, 2),
    }
    labels = {r["id"]: r.get("raw_label") for r in manifest["rooms"]}
    for r in rooms:
        pid = f"P{r['rid']:03d}"
        r["id"] = pid
        r["matched_track_a"] = p2g.get(pid)
        r["track_a_label"] = labels.get(p2g.get(pid))
    gt_doors = [o for o in manifest["openings"] if o["kind"] == "door"]
    pr_doors = [o for o in openings if o["kind"] == "door"]
    dm = match_points([o["position"] for o in pr_doors], [o["position"] for o in gt_doors], 1000.0)
    werr = [abs(pr_doors[p]["width_mm"] - gt_doors[g]["width_mm"]) for g, p in dm]
    door_metrics = {"gt_doors": len(gt_doors), "pred_doors": len(pr_doors), "matched_within_1m": len(dm),
                    "recall": round(len(dm) / max(len(gt_doors), 1), 4), "precision": round(len(dm) / max(len(pr_doors), 1), 4),
                    "width_mae_mm": round(float(np.mean(werr)), 1) if werr else None,
                    "width_median_abs_err_mm": round(float(np.median(werr)), 1) if werr else None}
    gt_wall = {tuple(sorted((g2p[e["a"]], g2p[e["b"]]))) for e in manifest["adjacency"]
               if e.get("shared_wall_mm") and e["a"] in g2p and e["b"] in g2p}
    pr_wall = {tuple(sorted((e["a"], e["b"]))) for e in adjacency if e["a"] in p2g and e["b"] in p2g}
    gt_door = {tuple(sorted((g2p[e["a"]], g2p[e["b"]]))) for e in manifest["adjacency"]
               if (e.get("doors") or e.get("passages")) and e["a"] in g2p and e["b"] in g2p}
    pr_door = {tuple(sorted(o["connects"])) for o in pr_doors if len(o["connects"]) == 2 and all(c in p2g for c in o["connects"])}
    return {"rooms": room_metrics, "doors": door_metrics,
            "adjacency_wall": edge_f1(pr_wall, gt_wall), "adjacency_door": edge_f1(pr_door, gt_door)}


def overlay(img, pred, rooms=None, room_lab=None, alpha=0.45):
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    col = colorize(pred)[:, :, ::-1]
    m = pred > 0
    base[m] = (base[m] * (1 - alpha) + col[m] * alpha).astype(np.uint8)
    if rooms is not None:
        rng = np.random.default_rng(1)
        for r in rooms:
            x, y, bw, bh = r["bbox_px"]
            sub = room_lab[y:y + bh, x:x + bw] == r["rid"]
            c = rng.integers(120, 255, 3)
            patch = base[y:y + bh, x:x + bw]
            patch[sub] = (patch[sub] * 0.6 + c * 0.4).astype(np.uint8)
    return base


def extract_structure(pred, conf, region, s):
    t0 = time.time()
    sealed, plug = seal_walls(pred, s)
    rooms, room_lab, exterior = extract_rooms(sealed, s, plug=plug)
    rooms = room_polygons(room_lab, rooms, region, s)
    openings = extract_openings(pred, conf, room_lab, exterior, region, s)
    adjacency = room_adjacency(room_lab, rooms, s)
    for r in rooms:
        r["id"] = f"P{r['rid']:03d}"
    door_edges = {}
    for o in openings:
        if len(o["connects"]) == 2:
            key = tuple(sorted(o["connects"]))
            door_edges.setdefault(key, {"doors": 0, "windows": 0})[o["kind"] + "s"] += 1
    for e in adjacency:
        e.update(door_edges.pop(tuple(sorted((e["a"], e["b"]))), {"doors": 0, "windows": 0}))
    for (a, b), v in door_edges.items():
        adjacency.append({"a": a, "b": b, "shared_wall_mm": 0.0, **v})
    return {"rooms": rooms, "room_lab": room_lab, "exterior": exterior, "sealed": sealed, "openings": openings,
            "adjacency": adjacency, "seconds": round(time.time() - t0, 1)}


def ocr_room_labels(dxf_path, region, rooms, s, ocr_mm_per_px=5.0, gpu=True, min_conf=0.3):
    import easyocr
    reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    img, _ = render_dxf(dxf_path, region, ocr_mm_per_px, line_px=1, with_text=True)
    f = s / ocr_mm_per_px
    for r in rooms:
        x, y, bw, bh = r["bbox_px"]
        crop = img[int(y * f):int((y + bh) * f), int(x * f):int((x + bw) * f)]
        if crop.size == 0 or min(crop.shape) < 16:
            r["ocr_text"] = ""
            continue
        res = reader.readtext(crop, detail=1, paragraph=False)
        words = [t for _, t, c in sorted(res, key=lambda q: (q[0][0][1], q[0][0][0])) if c >= min_conf]
        r["ocr_text"] = " ".join(words)
    return rooms


def label_agreement(rooms):
    def norm(t):
        return re.sub(r"[^A-Z0-9]", "", str(t or "").upper())
    rows = [r for r in rooms if r.get("track_a_label")]
    hit = [norm(r["track_a_label"]) and norm(r["track_a_label"]) in norm(r.get("ocr_text")) for r in rows]
    return {"rooms_with_track_a_label": len(rows), "ocr_contains_label": int(sum(hit)),
            "rate": round(float(np.mean(hit)), 4) if rows else None}


def to_manifest(struct, region, s, source, model_info):
    rooms = [{"id": r["id"], "polygon": r["polygon"], "area_m2": r["area_m2"], "enclosure": r["enclosure"],
              "matched_track_a": r.get("matched_track_a"), "track_a_label": r.get("track_a_label"),
              "ocr_text": r.get("ocr_text")} for r in struct["rooms"]]
    openings = [{"id": f"O{i + 1:03d}", **o} for i, o in enumerate(struct["openings"])]
    return {"schema": "trackC-raster-0.1", "source": source, "units": "mm", "region_mm": list(map(float, region)),
            "mm_per_px": s, "model": model_info, "rooms": rooms, "openings": openings, "adjacency": struct["adjacency"]}