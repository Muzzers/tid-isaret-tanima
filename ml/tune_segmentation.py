"""Surekli tanimadaki segment sinirlarini (onceki/sonraki kare sayisi) verideki kliplerle simule ederek secer.

Canli akis ~9-10 FPS; AUTSL 30 FPS. Klip seyreltilir (stride 3), etkinlik (el omuz ortasinin 1 omuz genisligi
altindan yukarida) kriteri SegmentTracker.kt ile aynidir: normalize edilmis koordinatlarda bilek y < 1.0.
Segment = ilk etkin kare - pre ... son etkin kare + post. Dogruluk validation uzerinde olculur.
"""
import argparse
import itertools

import numpy as np
import torch

import csv
from pathlib import Path

from features import ROOT, featurize, fix_length, to_vec
from models import load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="gru226")
ap.add_argument("--split", default="val")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

model, ck = load_checkpoint(a.name)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(dev)

rows = list(csv.reader(open(ROOT / "data" / "archive (1)" / f"{a.split}_labels.csv")))
clips, ys = [], []
for name, lab in rows:
    f = Path(ROOT / "data" / "keypoints" / a.split / f"{name}.npz")
    if f.exists():
        clips.append(featurize(f)); ys.append(int(lab))
    if a.limit and len(clips) >= a.limit:
        break
ys = np.array(ys)
print(a.split, len(clips), "klip")


def active_mask(h, p, k):
    wrist_y = np.where(k > 0.5, h[:, :, 0, 1], 9.0)  # [T,2], el yoksa buyuk (etkin degil)
    return (wrist_y < 1.0).any(1)


def segment(clip, stride, pre, post):
    h, p, k = clip
    idx = np.arange(0, len(h), stride)
    h, p, k = h[idx], p[idx], k[idx]
    act = active_mask(h, p, k)
    starts = [i for i in range(len(act) - 1) if act[i] and act[i + 1]]
    if not starts:
        return None
    s = starts[0]
    e = int(np.where(act)[0][-1])
    s0 = max(0, s - pre)
    e0 = min(len(h) - 1, e + post)
    if e0 - s0 + 1 < 12:
        return None
    return h[s0:e0 + 1], p[s0:e0 + 1], k[s0:e0 + 1]


@torch.no_grad()
def acc(segs):
    ok = [i for i, s in enumerate(segs) if s is not None]
    xs = np.stack([to_vec(*fix_length(*segs[i])) for i in ok])
    out = []
    for i in range(0, len(xs), 512):
        out.append(model(torch.tensor(xs[i:i + 512]).to(dev)).cpu().numpy())
    lg = np.concatenate(out)
    pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    c = pr.max(1)
    pred = pr.argmax(1)
    y = ys[ok]
    return len(ok) / len(segs), (pred == y).mean(), ((c >= 0.6) & (pred == y)).sum() / len(segs), (c >= 0.6).mean()


print("stride pre post | segment bulunan | dogruluk | esik0.6: dogru kabul orani (tum kliplere gore) | kabul orani")
for stride, pre, post in itertools.product([1, 3], [0, 2, 4, 8], [0, 2, 5]):
    segs = [segment(c, stride, pre, post) for c in clips]
    cov, ac, good, accepted = acc(segs)
    print(f"{stride}      {pre}   {post}   | {cov:.3f}           | {ac:.3f}    | {good:.3f}                       | {accepted:.3f}")

full = [(c[0][::3], c[1][::3], c[2][::3]) for c in clips]
_, ac, good, accepted = acc(full)
print(f"referans: tum klip, stride 3: dogruluk {ac:.3f}, esik0.6 dogru kabul {good:.3f}, kabul {accepted:.3f}")
full = [c for c in clips]
_, ac, good, accepted = acc(full)
print(f"referans: tum klip, stride 1: dogruluk {ac:.3f}, esik0.6 dogru kabul {good:.3f}, kabul {accepted:.3f}")
