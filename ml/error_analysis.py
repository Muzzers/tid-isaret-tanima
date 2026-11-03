"""Kayitli bir model uzerinden hata analizi, sinif bazli F1 ve grafik uretimi.

Vize raporundaki "Hata Analizi" bolumu icin:
  - Test kumesinde karisiklik matrisi (confusion matrix)
  - En cok birbiriyle karisan 10 isaret cifti (orn. abla <-> agabey)
  - F1 skoru en dusuk 10 sinif
  - Egitim kayip / dogrulama egri grafigi (experiments JSON'da history varsa)

Ornek:
    python ml/error_analysis.py --name gru_all_s0
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_gru import CNNTransformerNet, GRUNet, ROOT, batches, load_split

HERE = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"


def load_sign_names(csv_path):
    names = {}
    if not Path(csv_path).exists():
        return names
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names[int(row["ClassId"])] = row["TR"]
    return names


@torch.no_grad()
def predict_all(model, X, y, dev):
    model.eval()
    preds, trues = [], []
    for xb, yb in batches(X, y, 128, np.random.default_rng(0), False):
        p = model(xb.to(dev)).argmax(1).cpu().numpy()
        preds.append(p)
        trues.append(yb.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def compute_metrics(y_true, y_pred, n_cls):
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = []
    for i in range(n_cls):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        sup = int(cm[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class.append(dict(idx=i, support=sup, precision=round(prec, 4),
                              recall=round(rec, 4), f1=round(f1, 4)))
    return cm, per_class


def plot_curves(exp_files, out_path):
    loaded = []
    for p in exp_files:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        if d.get("history"):
            loaded.append((d.get("model", Path(p).stem), d["history"]))
    if not loaded:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for label, hist in loaded:
        eps = [h["ep"] for h in hist]
        ax1.plot(eps, [h["loss"] for h in hist], label=label, linewidth=1.8)
        ax2.plot(eps, [h["val"] * 100 for h in hist], label=label, linewidth=1.8)

    ax1.set_title("Eğitim Kaybı (Cross-Entropy)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_title("Doğrulama Doğruluğu (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def plot_top_confusion(cm, classes, sign_names, out_path, top_k=20):
    # Hataya en cok karisan top_k sinifin alt-matrisi (226x226 kagitta okunmuyor)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    err_per_cls = off.sum(axis=1) + off.sum(axis=0)
    k = min(top_k, len(classes))
    idx = np.argsort(err_per_cls)[::-1][:k]
    idx = np.sort(idx)
    sub = cm[np.ix_(idx, idx)]
    labels = [f"{sign_names.get(classes[i], str(classes[i]))}" for i in idx]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(sub, cmap="Blues")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Tahmin Edilen İşaret")
    ax.set_ylabel("Gerçek İşaret")
    ax.set_title(f"En Çok Karışan {k} Sınıf İçin Karışıklık Matrisi")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gru226", help="checkpoints/<name>.pt dosya adi")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--kp", default=str(ROOT / "data" / "keypoints"))
    ap.add_argument("--data", default=str(ROOT / "data" / "archive (1)"))
    ap.add_argument("--signs", default=str(ROOT / "docs" / "SignList_ClassId_TR_EN.csv"))
    ap.add_argument("--curves", default="", help="egri grafigi icin virgullu experiments/*.json listesi")
    a = ap.parse_args()

    ck_path = ROOT / "checkpoints" / f"{a.name}.pt"
    ck = torch.load(ck_path, map_location="cpu")
    classes = ck["classes"]
    label_map = {c: i for i, c in enumerate(classes)}
    sign_names = load_sign_names(a.signs)

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    if ck.get("model_type") == "cnn_transformer":
        model = CNNTransformerNet(ck["n_classes"])
    else:
        model = GRUNet(ck["n_classes"], hidden=ck["hidden"])
    model.load_state_dict(ck["state"])
    model = model.to(dev)

    X, y = load_split(a.kp, a.split, label_map, Path(a.data) / f"{a.split}_labels.csv")
    print(f"{a.split}: {len(X)} ornek yuklendi ({ck.get('arch', a.name)})")
    y_true, y_pred = predict_all(model, X, y, dev)

    cm, per_class = compute_metrics(y_true, y_pred, len(classes))
    for item in per_class:
        cid = classes[item["idx"]]
        item["class_id"] = cid
        item["name"] = sign_names.get(cid, f"sinif_{cid}")

    acc = float((y_true == y_pred).mean())
    macro_f1 = float(np.mean([c["f1"] for c in per_class if c["support"] > 0]))

    # En dusuk F1'li 10 sinif
    worst_10 = sorted([c for c in per_class if c["support"] > 0], key=lambda x: (x["f1"], x["recall"]))[:10]

    # En cok karisan 10 sinif cifti (diyagonal disi)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat_idx = np.argsort(off.ravel())[::-1][:10]
    confused_pairs = []
    for fi in flat_idx:
        i, j = divmod(int(fi), len(classes))
        cnt = int(off[i, j])
        if cnt == 0:
            break
        ci, cj = classes[i], classes[j]
        confused_pairs.append(dict(
            true_id=ci, true_name=sign_names.get(ci, str(ci)),
            pred_id=cj, pred_name=sign_names.get(cj, str(cj)),
            count=cnt, true_total=int(cm[i].sum())
        ))

    EXPERIMENTS.mkdir(exist_ok=True)
    cm_png = EXPERIMENTS / f"cm_{a.name}.png"
    plot_top_confusion(cm, classes, sign_names, cm_png)

    curve_files = [x.strip() for x in a.curves.split(",") if x.strip()]
    if not curve_files:
        curve_files = sorted(str(p) for p in EXPERIMENTS.glob(f"*_{a.name}.json"))
    curve_png = EXPERIMENTS / f"curve_{a.name}.png"
    has_curve = plot_curves(curve_files, curve_png)

    out_json = EXPERIMENTS / f"error_{a.name}.json"
    out_json.write_text(json.dumps(dict(
        name=a.name, arch=ck.get("arch", a.name), split=a.split, n_samples=len(y_true),
        accuracy=round(acc, 4), macro_f1=round(macro_f1, 4),
        worst_10=worst_10, confused_pairs=confused_pairs
    ), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nGenel Dogruluk: %{acc * 100:.2f} | Macro F1: %{macro_f1 * 100:.2f}")
    print("\n### En Çok Karışan 10 İşaret Çifti")
    print("| Gerçek İşaret | Tahmin Edilen | Adet / Toplam |")
    print("|---|---|---|")
    for p in confused_pairs:
        print(f"| {p['true_name']} ({p['true_id']}) | {p['pred_name']} ({p['pred_id']}) | {p['count']} / {p['true_total']} |")

    print("\n### F1 Skoru En Düşük 10 Sınıf")
    print("| Sınıf | Precision | Recall | F1 | Destek |")
    print("|---|---|---|---|---|")
    for c in worst_10:
        print(f"| {c['name']} ({c['class_id']}) | %{c['precision']*100:.1f} | %{c['recall']*100:.1f} | %{c['f1']*100:.1f} | {c['support']} |")

    print(f"\nKaydedildi: {out_json} | {cm_png}" + (f" | {curve_png}" if has_curve else ""))


if __name__ == "__main__":
    main()

