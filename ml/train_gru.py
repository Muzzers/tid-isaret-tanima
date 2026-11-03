"""10 siniflik (veya verilen alt kume) GRU temel modeli.

Girdi: extract_keypoints.py ciktisi (.npz). Normalizasyon PLAN.md / PDF denklem 1:
omuz ortasina oteleme + omuz genisligine bolme. Test kumesi yalnizca egitim sonunda bir kez olculur.

--model cnn_transformer ile PLAN.md IP3'teki karsilastirma modeli (1D-CNN + kucuk Transformer)
egitilir; ayni veri, ayni artirma ve ayni degerlendirme kullanilir ki fark sadece mimariden gelsin.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
T_FIXED = 64


def normalize(d):
    hands, pose = d["hands"].copy(), d["pose"].copy()
    hok, pok = d["hand_ok"], d["pose_ok"]
    sh_l, sh_r = pose[:, 1, :2], pose[:, 2, :2]  # sol/sag omuz
    center = (sh_l + sh_r) / 2
    scale = np.linalg.norm(sh_l - sh_r, axis=1)
    # omuz bulunamayan kareler: son gecerli degeri tasi, yoksa 1
    good = pok & (scale > 1e-4)
    if not good.any():
        center[:] = 0.5
        scale[:] = 0.3
    else:
        idx = np.where(good, np.arange(len(good)), 0)
        np.maximum.accumulate(idx, out=idx)
        first = np.argmax(good)
        idx[: first + 1] = first
        center, scale = center[idx], scale[idx]
    hands = (hands[..., :2] - center[:, None, None, :]) / scale[:, None, None, None]
    pose = (pose[..., :2] - center[:, None, :]) / scale[:, None, None]
    hands = hands * hok[:, :, None, None]  # eksik el = 0
    return hands.astype(np.float32), pose.astype(np.float32), hok.astype(np.float32)


def resample(x, t_out):
    t = len(x)
    idx = np.linspace(0, t - 1, t_out)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (idx - lo).reshape(-1, *([1] * (x.ndim - 1)))
    return x[lo] * (1 - w) + x[hi] * w


def fix_length(hands, pose, hok):
    return (resample(hands, T_FIXED).astype(np.float32), resample(pose, T_FIXED).astype(np.float32),
            (resample(hok, T_FIXED) > 0.5).astype(np.float32))


def featurize(path):
    """Normalize edilmis ham dizi (degisken uzunluk); sabit uzunluga batch sirasinda getirilir."""
    return normalize(np.load(path))  # [T,2,21,2], [T,7,2], [T,2]


def to_vec(hands, pose, hok):
    t = len(hands)
    v = np.concatenate([hands.reshape(t, -1), pose.reshape(t, -1), hok], axis=1)  # 84+14+2=100
    return v.astype(np.float32)


def load_split(kp_dir, split, label_map, labels_csv):
    X, y = [], []
    for name, lab in csv.reader(open(labels_csv)):
        lab = int(lab)
        if lab not in label_map:
            continue
        f = Path(kp_dir) / split / f"{name}.npz"
        if f.exists():
            X.append(featurize(f))
            y.append(label_map[lab])
    return X, np.array(y)


def augment(hands, pose, hok, rng):
    """Egitimde: zamansal seyreltme (telefonun dusuk kare hizini taklit eder), kirpma,
    rastgele olcek, +-10 derece donme, kaydirma."""
    t = len(hands)
    stride = int(rng.choice([1, 1, 2, 3]))  # 30 FPS / 15 FPS / 10 FPS
    idx = np.arange(int(rng.integers(0, stride)), t, stride)
    if len(idx) >= 8:
        hands, pose, hok = hands[idx], pose[idx], hok[idx]
    t = len(hands)
    cut = int(t * 0.1)
    t0 = int(rng.integers(0, cut + 1)); t1 = t - int(rng.integers(0, cut + 1))
    hands, pose, hok = fix_length(hands[t0:t1], pose[t0:t1], hok[t0:t1])
    a = np.deg2rad(rng.uniform(-10, 10))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    s = rng.uniform(0.9, 1.1)
    shift = rng.normal(0, 0.05, 2).astype(np.float32)
    hands = (hands @ R.T) * s + shift
    pose = (pose @ R.T) * s + shift
    hands = hands * hok[:, :, None, None]
    return hands, pose, hok


class GRUNet(nn.Module):
    def __init__(self, n_cls, in_dim=100, hidden=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.out = nn.Linear(hidden * 2, n_cls)

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        return self.out(h.mean(1))


class PositionalEncoding(nn.Module):
    """Transformer zamansal sirayi kendisi gormedigi icin sinus tabanli konum ekliyoruz."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CNNTransformerNet(nn.Module):
    """1D-CNN ile yerel hareket ozellikleri, ardindan kucuk Transformer ile zaman iliskileri.

    GRU'ya gore fark: tekrarlayan katman yok, egitim paralellestirilebilir. Model yine de
    hafif kalsin diye d_model=256 ve 2 katman secildi (~1,5 M parametre).
    """

    def __init__(self, n_cls, in_dim=100, d_model=256, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 128, 5, padding=2), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(128, d_model, 3, padding=1, stride=2), nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.BatchNorm1d(d_model), nn.GELU(),
        )
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 2, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, n_cls)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, T/2, d_model]
        h = self.transformer(self.pos(h))
        return self.out(self.norm(h.mean(1)))


def batches(X, y, bs, rng, train):
    order = rng.permutation(len(X)) if train else np.arange(len(X))
    for i in range(0, len(X), bs):
        ids = order[i : i + bs]
        vecs = []
        for j in ids:
            h, p, k = X[j]
            h, p, k = augment(h, p, k, rng) if train else fix_length(h, p, k)
            vecs.append(to_vec(h, p, k))
        yield torch.tensor(np.stack(vecs)), torch.tensor(y[ids])


@torch.no_grad()
def evaluate(model, X, y, dev):
    model.eval()
    correct = 0
    for xb, yb in batches(X, y, 128, np.random.default_rng(0), False):
        correct += (model(xb.to(dev)).argmax(1).cpu() == yb).sum().item()
    return correct / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="all", help="virgullu sinif no listesi veya 'all' (226)")
    ap.add_argument("--model", choices=["gru", "cnn_transformer"], default="gru",
                    help="gru = BiGRU temel model, cnn_transformer = karsilastirma modeli")
    ap.add_argument("--kp", default=str(ROOT / "data" / "keypoints"))
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--data", default=str(ROOT / "data" / "archive (1)"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="gru226")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    classes = list(range(226)) if a.classes == "all" else [int(c) for c in a.classes.split(",")]
    label_map = {c: i for i, c in enumerate(classes)}
    data = {s: load_split(a.kp, s, label_map, Path(a.data) / f"{s}_labels.csv") for s in ("train", "val", "test")}
    for s, (X, y) in data.items():
        print(s, len(X), flush=True)

    # Mac'te MPS, egitim makinesinde CUDA; ikisi de yoksa CPU.
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    if a.model == "gru":
        model = GRUNet(len(classes), hidden=a.hidden)
        arch = f"BiGRU-2x{a.hidden}"
    else:
        model = CNNTransformerNet(len(classes))
        arch = "1D-CNN+Transformer"
    model = model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {arch} | parametre {n_params / 1e6:.2f} M | cihaz {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best, best_state = 0.0, None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in batches(*data["train"], a.batch, rng, True):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb.to(dev)), yb.to(dev), label_smoothing=0.1)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        sched.step()
        va = evaluate(model, *data["val"], dev)
        if va > best:
            best, best_state = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"ep {ep:02d} loss {tot / len(data['train'][1]):.3f} val {va:.3f}", flush=True)

    model.load_state_dict(best_state)
    ck = ROOT / "checkpoints"
    ck.mkdir(exist_ok=True)
    torch.save(dict(state=best_state, hidden=a.hidden, n_classes=len(classes), classes=classes, t_fixed=T_FIXED,
                    model_type=a.model, arch=arch), ck / f"{a.name}.pt")
    test = evaluate(model, *data["test"], dev)
    print(f"EN IYI val {best:.3f} | TEST {test:.3f} | sure {time.time() - t0:.0f}s")
    out = ROOT / "experiments"
    out.mkdir(exist_ok=True)
    (out / f"{time.strftime('%Y%m%d')}_{a.name}.json").write_text(
        json.dumps(dict(model=arch, model_type=a.model, params=n_params, n_classes=len(classes),
                        classes=classes if len(classes) < 30 else "all", epochs=a.epochs, seed=a.seed, lr=a.lr,
                        n_train=len(data["train"][1]), best_val=best, test=test, device=str(dev),
                        duration_s=round(time.time() - t0)), indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
