"""Egitilmis modeli test kumesinde calistirip ornek tahminleri gosterir (sunum / hizli kontrol icin).

Rapordaki hattin tamami tek ekranda: nokta dosyasi -> normalizasyon -> 64 x 100 girdi -> model ->
kelime. Sonda tum test kumesindeki dogruluk da yazilir; bu sayi experiments/ altindaki JSON ile ayni cikmali.

Ornek:
    python ml/demo.py --name cnn_transformer_all_s0
    python ml/demo.py --name gru_all_s0 --n 20 --seed 3
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from features import ROOT, fix_length, load_sign_names, load_split, to_vec
from models import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cnn_transformer_all_s0", help="checkpoints/<name>.pt dosya adi")
    ap.add_argument("--n", type=int, default=10, help="ekrana yazilacak ornek sayisi")
    ap.add_argument("--seed", type=int, default=0, help="hangi orneklerin secilecegi")
    ap.add_argument("--kp", default=str(ROOT / "data" / "keypoints"))
    ap.add_argument("--data", default=str(ROOT / "data" / "archive (1)"))
    ap.add_argument("--signs", default=str(ROOT / "docs" / "SignList_ClassId_TR_EN.csv"))
    a = ap.parse_args()

    model, ck = load_checkpoint(a.name)
    classes = ck["classes"]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {ck.get('arch', a.name)} | {n_params:,} parametre | {len(classes)} sinif".replace(",", "."))

    label_map = {c: i for i, c in enumerate(classes)}
    X, y = load_split(a.kp, "test", label_map, Path(a.data) / "test_labels.csv")
    names = load_sign_names(a.signs)

    # Tum test kumesi icin olasiliklar. Model kucuk oldugu icin CPU'da birkac saniye suruyor.
    vecs = np.stack([to_vec(*fix_length(*x)) for x in X])
    print(f"test: {len(X)} video | ilk videonun ham uzunlugu {len(X[0][0])} kare | model girdisi {vecs.shape[1:]}")
    with torch.no_grad():
        prob = torch.cat([model(torch.tensor(vecs[i : i + 256])).softmax(1) for i in range(0, len(vecs), 256)])
    conf, pred = prob.max(1)
    pred, conf = pred.numpy(), conf.numpy()

    print(f"\n{'gercek':<16}{'tahmin':<16}{'guven':>6}")
    for i in np.random.default_rng(a.seed).choice(len(y), a.n, replace=False):
        t, p = names[classes[y[i]]], names[classes[pred[i]]]
        print(f"{t:<16}{p:<16}{conf[i] * 100:>5.0f}%  {'dogru' if pred[i] == y[i] else 'YANLIS'}")

    print(f"\ntest dogrulugu (Top-1): %{(pred == y).mean() * 100:.2f}  ({(pred == y).sum()} / {len(y)})")


if __name__ == "__main__":
    main()
