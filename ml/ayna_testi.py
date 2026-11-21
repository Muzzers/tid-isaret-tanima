"""Modellerin AUTSL test kumesinde normal ve ayna goruntusundeki dogrulugu.

Neden: AUTSL videolari aynali (Kinect), web kamerasi ve telefonun analiz goruntusu aynasiz. Bunu iki
veri setinde baskin elin hangi tarafta goründugunden anladik (AUTSL'de videolarin %94'unde govde
modelinin "sol" bilegi, ASL Citizen'da %92'sinde "sag" bilegi daha hizli). Aynaya duyarli bir model
telefonda isaretleri ters gorur. Egitimde --ayna ile bu fark kapatilir.

Ornek:
    python ml/ayna_testi.py --models cnn_transformer_all_s1,cnn_transformer_ayna_s0
"""
import argparse
import csv

import numpy as np
import torch

from features import ROOT, aynala, fix_length, normalize, to_vec
from models import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="cnn_transformer_all_s1")
    a = ap.parse_args()

    ham, y = [], []
    for name, lab in csv.reader(open(ROOT / "data" / "archive (1)" / "test_labels.csv")):
        f = ROOT / "data" / "keypoints" / "test" / f"{name}.npz"
        if f.exists():
            ham.append(normalize(dict(np.load(f))))
            y.append(int(lab))
    y = np.array(y)
    girdiler = {
        "normal": np.stack([to_vec(*fix_length(*x)) for x in ham]),
        "aynali": np.stack([to_vec(*fix_length(*aynala(*x))) for x in ham]),
    }

    print(f"{'model':<28}{'normal':>10}{'aynali':>10}   (test dogrulugu, {len(y)} video)")
    for ad in a.models.split(","):
        model, ck = load_checkpoint(ad)
        sinif = np.array(ck["classes"])
        sonuc = []
        for v in girdiler.values():
            with torch.no_grad():
                p = torch.cat([model(torch.tensor(v[i : i + 512])).argmax(1) for i in range(0, len(v), 512)])
            sonuc.append((sinif[p.numpy()] == y).mean() * 100)
        print(f"{ad:<28}{sonuc[0]:>9.2f}%{sonuc[1]:>9.2f}%")


if __name__ == "__main__":
    main()
