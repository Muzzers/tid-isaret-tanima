"""PyTorch checkpoint -> numpy agirliklari + test girdileri/referans ciktilari (TF/LiteRT donusumu icin).

ml/.venv (torch) ile calisir. Cikti: checkpoints/<ad>_weights.npz ve data/export/<ad>_test.npz
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from train_gru import CNNTransformerNet, GRUNet, ROOT, batches, load_split

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="gru226")
ap.add_argument("--n_test", type=int, default=0, help="0 = tum test seti")
a = ap.parse_args()

ck = torch.load(ROOT / "checkpoints" / f"{a.name}.pt", map_location="cpu")
if ck.get("model_type") == "cnn_transformer":
    model = CNNTransformerNet(ck["n_classes"])
else:
    model = GRUNet(ck["n_classes"], hidden=ck["hidden"])
model.load_state_dict(ck["state"])
model.eval()
w = {k: v.numpy() for k, v in ck["state"].items()}
np.savez(ROOT / "checkpoints" / f"{a.name}_weights.npz", hidden=ck["hidden"], **{k.replace(".", "__"): v for k, v in w.items()})

classes = ck["classes"]
label_map = {c: i for i, c in enumerate(classes)}
X, y = load_split(ROOT / "data" / "keypoints", "test", label_map, ROOT / "data" / "archive (1)" / "test_labels.csv")
if a.n_test:
    X, y = X[: a.n_test], y[: a.n_test]
xs = []
for xb, _ in batches(X, y, 256, np.random.default_rng(0), False):
    xs.append(xb.numpy())
x = np.concatenate(xs)
with torch.no_grad():
    ref = model(torch.tensor(x)).numpy()
out = ROOT / "data" / "export"
out.mkdir(parents=True, exist_ok=True)
np.savez(out / f"{a.name}_test.npz", x=x, y=y, ref=ref)
print("test", x.shape, "torch dogruluk", (ref.argmax(1) == y).mean())
