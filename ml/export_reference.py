"""Uygulamadaki isaret listesi icin her kelimenin ornek hareketini disa aktarir.

Her sinif icin egitim kumesinden bir video secilir: modelin dogru ve en emin bildigi, iki elin de
cogu karede gorundugu ornek. Noktalar egitimdeki gibi normalize edilir (omuz ortasi = 0, omuz
genisligi = 1), 48 kareye ornekleniyor ve int16 olarak (deger x 1000) yaziliyor.

Cikti: app/assets/reference.bin
    baslik: 'TIDR', sinif sayisi, kare sayisi, kare basina deger sayisi (int32)
    govde : [sinif][kare][100] int16  (84 el + 14 govde + 2 el-var bayragi; siralama to_vec ile ayni)

Ornek:
    python ml/export_reference.py --name gru226
"""
import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from features import ROOT, fix_length, load_split, resample, to_vec
from models import load_checkpoint

FRAMES = 48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gru226", help="ornek secimi icin kullanilacak model")
    ap.add_argument("--kp", default=str(ROOT / "data" / "keypoints"))
    ap.add_argument("--data", default=str(ROOT / "data" / "archive (1)"))
    ap.add_argument("--out", default=str(ROOT / "app" / "assets" / "reference.bin"))
    a = ap.parse_args()

    model, ck = load_checkpoint(a.name)
    classes = ck["classes"]
    label_map = {c: i for i, c in enumerate(classes)}
    X, y = load_split(a.kp, "train", label_map, Path(a.data) / "train_labels.csv")
    print(f"egitim: {len(X)} video")

    vecs = np.stack([to_vec(*fix_length(*x)) for x in X])
    with torch.no_grad():
        prob = torch.cat([model(torch.tensor(vecs[i : i + 512])).softmax(1) for i in range(0, len(vecs), 512)]).numpy()

    out = np.zeros((len(classes), FRAMES, 100), np.int16)
    for c in range(len(classes)):
        ids = np.where(y == c)[0]
        hand_ratio = np.array([X[i][2].mean() for i in ids])  # iki elin gorunme orani (0-1)
        p_true = prob[ids, c]
        correct = prob[ids].argmax(1) == c
        score = p_true + 0.5 * hand_ratio + correct  # once dogru tahmin, sonra guven ve el gorunurlugu
        best = ids[int(np.argmax(score))]
        h, p, k = X[best]
        v = to_vec(resample(h, FRAMES), resample(p, FRAMES), (resample(k, FRAMES) > 0.5).astype(np.float32))
        out[c] = np.clip(np.round(v * 1000), -32000, 32000).astype(np.int16)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(b"TIDR" + struct.pack("<3i", len(classes), FRAMES, 100))
        f.write(out.astype("<i2").tobytes())
    xy = out[:, :, :98].reshape(-1, 2) / 1000
    xy = xy[np.abs(xy).sum(1) > 0]
    print(f"yazildi: {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB)")
    print("x araligi (%1-%99):", np.percentile(xy[:, 0], [1, 99]).round(2), " y araligi:", np.percentile(xy[:, 1], [1, 99]).round(2))


if __name__ == "__main__":
    main()
