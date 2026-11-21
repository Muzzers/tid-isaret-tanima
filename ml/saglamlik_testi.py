"""Canli kullanimdaki farklari AUTSL test kumesinde taklit edip modelin dogrulugunu olcer.

Kayitli veride %90 olan model telefonda cok daha kotu calisiyor. Telefon kaydi gelmeden hangi farkin
ne kadar zarar verdigini gormek icin test videolarina tek tek bozulma uygulanir:

  ayna         : goruntu aynali (AUTSL aynali, telefon aynasiz)
  asagidan     : kamera asagidan bakiyor -> dikey mesafeler kisalir (y * 0.75)
  yukaridan    : kamera yukaridan bakiyor (y * 1.25)
  bos_basta    : kayit erken basladi, basta %40 uzunlugunda eller asagida bekleme
  bos_sonda    : kayit gec bitti, sonda %40 bekleme
  tus_basta    : "Kaydet"e basan el basta kadraja yakin ve buyuk giriyor
  el_kaybi_30  : karelerin %30'unda el bulunamiyor (isik, hiz, bulaniklik)
  titreme      : noktalarda omuz genisliginin %4'u kadar gurultu
  omuz_kayik   : omuz merkezi yanlis bulunmus (0,3 omuz genisligi asagi)
  hizli / yavas: isaret 1,6 kat hizli ya da yavas (10 FPS'te)
  karma        : asagidan + bos_basta + bos_sonda + el_kaybi_30 + titreme + hizli birlikte

Ornek:
    python ml/saglamlik_testi.py --models cnn_transformer_asl_ayna_s0
"""
import argparse
import csv

import numpy as np
import torch

from features import ROOT, aynala, fix_length, normalize, to_vec
from models import load_checkpoint


def bekleme(pose, n, rng):
    """Eller asagida, kadrajda yok; govde sabit (son/ilk karedeki gibi)."""
    h = np.zeros((n, 2, 21, 2), np.float32)
    k = np.zeros((n, 2), np.float32)
    p = np.repeat(pose[None], n, 0) + rng.normal(0, 0.01, (n, 7, 2)).astype(np.float32)
    return h, p, k


def tus_eli(pose, n):
    """Sag el ekranin alt ortasindan kameraya dogru uzaniyor: buyuk ve asagida."""
    h = np.zeros((n, 2, 21, 2), np.float32)
    k = np.zeros((n, 2), np.float32)
    rng = np.random.default_rng(0)
    taban = rng.normal(0, 0.6, (21, 2)).astype(np.float32)  # buyuk el: omuz genisliginin ~2 kati
    for i in range(n):
        h[i, 1] = taban + np.array([0.3, 2.2 - 0.6 * i / max(n - 1, 1)], np.float32)
        k[i, 1] = 1
    p = np.repeat(pose[None], n, 0)
    return h, p, k


def boz(ad, h, p, k, rng):
    t = len(h)
    if ad == "normal":
        return h, p, k
    if ad == "karma":  # canli kullanimdaki farklarin birlikte gelmesi
        for alt in ("asagidan", "bos_basta", "bos_sonda", "el_kaybi_30", "titreme", "hizli"):
            h, p, k = boz(alt, h, p, k, rng)
        return h, p, k
    if ad == "ayna":
        return aynala(h, p, k)
    if ad in ("asagidan", "yukaridan"):
        c = 0.75 if ad == "asagidan" else 1.25
        h, p = h.copy(), p.copy()
        h[..., 1] *= c
        p[..., 1] *= c
        return h * k[:, :, None, None], p, k
    if ad in ("bos_basta", "bos_sonda"):
        n = max(int(t * 0.4), 1)
        bh, bp, bk = bekleme(p[0] if ad == "bos_basta" else p[-1], n, rng)
        if ad == "bos_basta":
            return np.concatenate([bh, h]), np.concatenate([bp, p]), np.concatenate([bk, k])
        return np.concatenate([h, bh]), np.concatenate([p, bp]), np.concatenate([k, bk])
    if ad == "tus_basta":
        n = max(int(t * 0.2), 1)
        th, tp, tk = tus_eli(p[0], n)
        return np.concatenate([th, h]), np.concatenate([tp, p]), np.concatenate([tk, k])
    if ad == "el_kaybi_30":
        k = k * (rng.random((t, 2)) > 0.3)
        return h * k[:, :, None, None], p, k
    if ad == "titreme":
        h = h + rng.normal(0, 0.04, h.shape).astype(np.float32)
        p = p + rng.normal(0, 0.04, p.shape).astype(np.float32)
        return h * k[:, :, None, None], p, k
    if ad == "omuz_kayik":
        kay = np.array([0, -0.3], np.float32)  # merkez asagida bulundu -> noktalar yukari kayar
        return (h + kay) * k[:, :, None, None], p + kay, k
    if ad in ("hizli", "yavas"):
        oran = 1.6 if ad == "hizli" else 1 / 1.6
        idx = np.round(np.arange(0, t, 3 * oran)).astype(int)  # 10 FPS
        idx = idx[idx < t]
        if len(idx) < 4:
            idx = np.arange(0, t, max(t // 4, 1))
        return h[idx], p[idx], k[idx]
    raise ValueError(ad)


KOSULLAR = ["normal", "ayna", "asagidan", "yukaridan", "bos_basta", "bos_sonda", "tus_basta",
            "el_kaybi_30", "titreme", "omuz_kayik", "hizli", "yavas", "karma"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="cnn_transformer_asl_ayna_s0")
    ap.add_argument("--kosullar", default=",".join(KOSULLAR))
    ap.add_argument("--limit", type=int, default=0, help="test kumesinden ilk N video (hizli deneme)")
    a = ap.parse_args()

    ham, y = [], []
    for name, lab in csv.reader(open(ROOT / "data" / "archive (1)" / "test_labels.csv")):
        f = ROOT / "data" / "keypoints" / "test" / f"{name}.npz"
        if f.exists():
            ham.append(normalize(dict(np.load(f))))
            y.append(int(lab))
        if a.limit and len(ham) >= a.limit:
            break
    y = np.array(y)
    kosullar = a.kosullar.split(",")
    girdiler = {}
    for ad in kosullar:
        rng = np.random.default_rng(0)
        girdiler[ad] = np.stack([to_vec(*fix_length(*boz(ad, *x, rng))) for x in ham])

    modeller = [(ad, *load_checkpoint(ad)) for ad in a.models.split(",")]
    print(f"{'kosul':<14}" + "".join(f"{m[0][-22:]:>24}" for m in modeller) + f"   ({len(y)} test videosu)")
    for ad in kosullar:
        satir = f"{ad:<14}"
        for _, model, ck in modeller:
            sinif = np.array(ck["classes"])
            with torch.no_grad():
                v = girdiler[ad]
                p = torch.cat([model(torch.tensor(v[i : i + 512])).argmax(1) for i in range(0, len(v), 512)])
            satir += f"{(sinif[p.numpy()] == y).mean() * 100:>23.2f}%"
        print(satir, flush=True)


if __name__ == "__main__":
    main()
