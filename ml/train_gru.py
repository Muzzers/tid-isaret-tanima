"""Egitim betigi: BiGRU (varsayilan) ya da --model cnn_transformer.

Iki model ayni veri, ayni artirma ve ayni degerlendirme ile egitilir ki fark sadece mimariden gelsin.
Test kumesi yalnizca egitim sonunda bir kez olculur. Veri hazirligi features.py, mimariler models.py icinde.

Yabanci veriyle on egitim (model yine yalnizca TID'i taniyacak sekilde):
    1) --dataset asl_citizen ile Amerikan Isaret Dili'nde egit (cikis katmani ASL kelimeleri)
    2) --init <on egitim adi> ile AUTSL'de devam et: cikis katmani disindaki agirliklar alinir,
       cikis katmani 226 TID kelimesi icin sifirdan kurulur. Test yine AUTSL test kumesidir.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from features import ROOT, T_FIXED, asl_citizen_glosses, batches, load_asl_citizen, load_split
from models import CNNTransformerNet, GRUNet, pick_device


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
    ap.add_argument("--dataset", choices=["autsl", "asl_citizen"], default="autsl",
                    help="asl_citizen yalnizca on egitim icin (Amerikan Isaret Dili)")
    ap.add_argument("--ayna", action="store_true", help="egitimde yarim olasilikla ayna goruntusu (solak isaretci)")
    ap.add_argument("--init", default="", help="on egitilmis agirlik (checkpoints/<ad>.pt); cikis katmani haric alinir")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    if a.dataset == "asl_citizen":
        classes = asl_citizen_glosses()
        label_map = {c: i for i, c in enumerate(classes)}
        data = {s: load_asl_citizen(s, label_map) for s in ("train", "val", "test")}
    else:
        classes = list(range(226)) if a.classes == "all" else [int(c) for c in a.classes.split(",")]
        label_map = {c: i for i, c in enumerate(classes)}
        data = {s: load_split(a.kp, s, label_map, Path(a.data) / f"{s}_labels.csv") for s in ("train", "val", "test")}
    for s, (X, y) in data.items():
        print(s, len(X), flush=True)

    dev = pick_device()
    if a.model == "gru":
        model = GRUNet(len(classes), hidden=a.hidden)
        arch = f"BiGRU-2x{a.hidden}"
    else:
        model = CNNTransformerNet(len(classes))
        arch = "1D-CNN+Transformer"
    if a.init:
        onceki = torch.load(ROOT / "checkpoints" / f"{a.init}.pt", map_location="cpu")["state"]
        simdiki = model.state_dict()
        alinan = {k: v for k, v in onceki.items() if k in simdiki and v.shape == simdiki[k].shape}
        model.load_state_dict(alinan, strict=False)
        print(f"on egitim: {a.init} -> {len(alinan)}/{len(simdiki)} agirlik alindi (cikis katmani yeni)", flush=True)
    model = model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {arch} | parametre {n_params / 1e6:.2f} M | cihaz {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best, best_state = 0.0, None
    history = []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in batches(*data["train"], a.batch, rng, True, a.ayna):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb.to(dev)), yb.to(dev), label_smoothing=0.1)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        sched.step()
        tr_loss = tot / len(data["train"][1])
        va = evaluate(model, *data["val"], dev)
        history.append(dict(ep=ep, loss=round(tr_loss, 4), val=round(va, 4)))
        if va > best:
            best, best_state = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"ep {ep:02d} loss {tr_loss:.3f} val {va:.3f}", flush=True)

    model.load_state_dict(best_state)
    ck = ROOT / "checkpoints"
    ck.mkdir(exist_ok=True)
    torch.save(dict(state=best_state, hidden=a.hidden, n_classes=len(classes), classes=classes, t_fixed=T_FIXED,
                    model_type=a.model, arch=arch), ck / f"{a.name}.pt")
    test = evaluate(model, *data["test"], dev) if len(data["test"][1]) else float("nan")
    print(f"EN IYI val {best:.3f} | TEST {test:.3f} | sure {time.time() - t0:.0f}s")
    out = ROOT / "experiments"
    out.mkdir(exist_ok=True)
    (out / f"{time.strftime('%Y%m%d')}_{a.name}.json").write_text(
        json.dumps(dict(model=arch, model_type=a.model, params=n_params, n_classes=len(classes),
                        classes=classes if len(classes) < 30 else "all", epochs=a.epochs, seed=a.seed, lr=a.lr,
                        dataset=a.dataset, init=a.init or None, ayna=a.ayna,
                        n_train=len(data["train"][1]), best_val=best, test=test, device=str(dev),
                        duration_s=round(time.time() - t0), history=history), indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
