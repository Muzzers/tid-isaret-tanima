"""Telefonda on isleme paritesi icin: ilk N test orneginin HAM noktalarini ve Python tahminlerini
ikili dosyaya yazar (app/android/app/src/main/assets/selftest.bin).

Her ornek: int T, int etiket, int python_tahmini, sonra T x [hands 2*21*2 float, pose 7*2 float,
hand_ok 2 float, pose_ok 1 float] (little-endian, noktalar 0-1 normalize goruntu koordinati).
"""
import argparse
import csv
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="gru226", help="telefondaki modelle ayni olmali (export_weights.py ciktisi)")
N = 200
ref = np.load(ROOT / "data" / "export" / f"{ap.parse_args().name}_test.npz")["ref"].argmax(1)
rows = [r for r in csv.reader(open(ROOT / "data" / "archive (1)" / "test_labels.csv"))]
out = bytearray()
n = 0
for name, lab in rows:
    f = ROOT / "data" / "keypoints" / "test" / f"{name}.npz"
    if not f.exists():
        continue
    if n >= N:
        break
    d = np.load(f)
    T = len(d["hands"])
    out += struct.pack("<iii", T, int(lab), int(ref[n]))
    frame = np.concatenate([
        d["hands"][..., :2].reshape(T, -1), d["pose"][..., :2].reshape(T, -1),
        d["hand_ok"].astype(np.float32), d["pose_ok"].astype(np.float32)[:, None]], axis=1).astype("<f4")
    out += frame.tobytes()
    n += 1
dst = ROOT / "app" / "android" / "app" / "src" / "main" / "assets" / "selftest.bin"
dst.write_bytes(bytes(out))
print(n, "ornek,", len(out) / 1e6, "MB, python dogrulugu", (ref[:n] == np.array([int(r[1]) for r in rows if (ROOT/'data'/'keypoints'/'test'/f'{r[0]}.npz').exists()][:n])).mean())
