"""Reddetme esigi kalibrasyonu: en yuksek softmax olasiligi (guven) icin dogruluk-kapsam egrisi.

Sozluk disi hareket icin gercek veri yok; sahte negatifler kullanilir (eller yok, kare karistirma,
gurultu). Esik doğrulama setinde secilir, test setinde raporlanir.
"""
import argparse

import numpy as np
import torch

from features import ROOT, batches, load_split
from models import load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="gru226")
ap.add_argument("--target_precision", type=float, default=0.95)
a = ap.parse_args()

model, ck = load_checkpoint(a.name)
label_map = {c: i for i, c in enumerate(ck["classes"])}


def logits_for(split):
    X, y = load_split(ROOT / "data" / "keypoints", split, label_map, ROOT / "data" / "archive (1)" / f"{split}_labels.csv")
    xs = np.concatenate([xb.numpy() for xb, _ in batches(X, y, 256, np.random.default_rng(0), False)])
    with torch.no_grad():
        return xs, y, model(torch.tensor(xs)).numpy()


def conf(lg):
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return p.max(1), p.argmax(1)


xv, yv, lv = logits_for("val")
xt, yt, lt = logits_for("test")
cv, pv = conf(lv)
ct, pt = conf(lt)

print("esik  kapsam(val) dogruluk(kabul, val) | kapsam(test) dogruluk(kabul, test)")
chosen = None
for th in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    mv, mt = cv >= th, ct >= th
    pr_v = (pv[mv] == yv[mv]).mean() if mv.any() else float("nan")
    pr_t = (pt[mt] == yt[mt]).mean() if mt.any() else float("nan")
    print(f"{th:.2f}  {mv.mean():.3f}        {pr_v:.3f}                 | {mt.mean():.3f}         {pr_t:.3f}")
    if chosen is None and pr_v >= a.target_precision:
        chosen = th
print("secilen esik (val'de kabul edilenlerde dogruluk >= %.2f olan en kucuk):" % a.target_precision, chosen)

# sahte negatifler: gercek test girdilerinden turetilir
rng = np.random.default_rng(0)
neg = {}
x0 = xt[:1000].copy()
nohand = x0.copy(); nohand[:, :, :84] = 0; nohand[:, :, 98:100] = 0          # eller yok
shuffled = x0.copy()
for i in range(len(shuffled)): shuffled[i] = shuffled[i][rng.permutation(64)]  # zaman karisik
noise = rng.normal(0, 1, x0.shape).astype(np.float32)                           # gurultu
still = np.repeat(x0[:, :1, :], 64, axis=1)                                     # tek kare tekrar (hareketsiz)
for name, x in [("eller yok", nohand), ("kareler karisik", shuffled), ("gurultu", noise), ("hareketsiz", still)]:
    with torch.no_grad():
        c, _ = conf(model(torch.tensor(x)).numpy())
    row = " ".join(f"%{(c >= th).mean() * 100:.0f}" for th in [0.5, 0.7, 0.8, 0.9])
    print(f"negatif '{name}': kabul orani (esik 0.5/0.7/0.8/0.9): {row}")
