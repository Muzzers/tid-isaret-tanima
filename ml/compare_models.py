"""IP3: BiGRU ile 1D-CNN+Transformer modelini ayni kosullarda karsilastirir.

Her (model, tohum) ikilisi icin train_gru.py ayri bir surec olarak calistirilir. Boylece
bir kosu cokerken digerleri etkilenmez ve bellek birikmez. Sonuc, experiments/ altina
ozet JSON olarak yazilir; ekrana da ara rapora yapistirilabilecek markdown tablo basar.

Ornek:
    python ml/compare_models.py --seeds 0,1,2 --epochs 60
    python ml/compare_models.py --models gru --seeds 0 --epochs 5 --classes 30,175,65   # hizli deneme
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXPERIMENTS = ROOT / "experiments"


def run_one(model, seed, a):
    tag = "all" if a.classes == "all" else "sub"
    name = f"{model}_{tag}_s{seed}"
    cmd = [sys.executable, str(HERE / "train_gru.py"), "--model", model, "--seed", str(seed),
           "--epochs", str(a.epochs), "--classes", a.classes, "--name", name,
           "--kp", a.kp, "--data", a.data, "--batch", str(a.batch), "--lr", str(a.lr)]
    # Ayni kosu daha once bittiyse tekrar egitme: uzun kosu yarida kesilirse kaldigi yerden
    # devam edilebilsin, eski tohumlar da ozete girsin. --force ile yeniden kosulur.
    for old in sorted(EXPERIMENTS.glob(f"*_{name}.json"), reverse=True):
        r = json.loads(old.read_text(encoding="utf-8"))
        # error_<ad>.json da ayni kaliba uyuyor; "epochs" alani olmayan dosya egitim kaydi degildir.
        if not a.force and (r.get("epochs"), r.get("seed"), r.get("lr")) == (a.epochs, seed, a.lr):
            print(f"\n=== {model} | seed {seed} | hazir: {old.name} ===", flush=True)
            r["wall_s"] = r["duration_s"]
            r["file"] = old.name
            return r
    print(f"\n=== {model} | seed {seed} | {a.epochs} epoch ===", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    # Dosya adindaki tarih kosunun bittigi gun; gece yarisini gecen kosuda bugunle tutmayabilir.
    out = max(EXPERIMENTS.glob(f"2*_{name}.json"), key=lambda p: p.stat().st_mtime)
    r = json.loads(out.read_text(encoding="utf-8"))
    r["wall_s"] = round(time.time() - t0)
    r["file"] = out.name
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gru,cnn_transformer")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--classes", default="all")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--kp", default=str(ROOT / "data" / "keypoints"))
    ap.add_argument("--data", default=str(ROOT / "data" / "archive (1)"))
    ap.add_argument("--force", action="store_true", help="bitmis kosulari da yeniden egit")
    a = ap.parse_args()

    models = [m.strip() for m in a.models.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    runs = [run_one(m, s, a) for m in models for s in seeds]

    # Ozet: model basina ortalama + standart sapma (aralik raporlamak icin)
    summary = {}
    for m in models:
        rs = [r for r in runs if r["model_type"] == m]
        summary[m] = dict(
            arch=rs[0]["model"],
            params=rs[0]["params"],
            seeds=[r["seed"] for r in rs],
            test=[round(r["test"], 4) for r in rs],
            val=[round(r["best_val"], 4) for r in rs],
            test_mean=round(statistics.mean(r["test"] for r in rs), 4),
            test_std=round(statistics.stdev(r["test"] for r in rs), 4) if len(rs) > 1 else 0.0,
            val_mean=round(statistics.mean(r["best_val"] for r in rs), 4),
            minutes_mean=round(statistics.mean(r["wall_s"] for r in rs) / 60, 1),
        )

    EXPERIMENTS.mkdir(exist_ok=True)
    dst = EXPERIMENTS / f"{time.strftime('%Y%m%d')}_compare_{a.classes.replace(',', '-')[:20]}.json"
    dst.write_text(json.dumps(dict(epochs=a.epochs, batch=a.batch, lr=a.lr, classes=a.classes,
                                   models=summary, runs=runs), indent=2), encoding="utf-8")

    print("\n| Model | Parametre | Seed | Test (ort. ± ss) | Val (ort.) | Süre (dk/koşu) |")
    print("|---|---|---|---|---|---|")
    for m, s in summary.items():
        pm = f"%{s['test_mean'] * 100:.2f} ± %{s['test_std'] * 100:.2f}"
        vm = f"%{s['val_mean'] * 100:.2f}"
        print(f"| {s['arch']} | {s['params'] / 1e6:.2f} M | {', '.join(map(str, s['seeds']))} | {pm} | {vm} | {s['minutes_mean']} |")
    print(f"\nozet: {dst}")


if __name__ == "__main__":
    main()
