"""Gonullu testi kayitlarini (uygulamadaki "Gonullu testi" ekrani) inceler.

Telefondan almak icin:
    adb pull /sdcard/Android/data/tr.edu.erciyes.tid_app/files/gonullu_testi runs/gonullu_testi

Yapilanlar:
  1. Her kaydi egitimle ayni on islemeden gecirip secilen modellerle yeniden siniflandirir
     (telefondaki tahminle karsilastirma: Kotlin ve Python ayni sonucu vermeli).
  2. Katilimci bazinda Top-1 / Top-3 dogruluk ve esigi gecenlerde dogruluk (tez onerisi Tablo 2).
  3. Canli kayitlari AUTSL test kumesiyle karsilastirir: kare sayisi, elin gorunme orani,
     kisinin kameraya uzakligi (omuz genisligi / goruntu genisligi), el hizlari. Canli dogruluk
     dusukse farkin nereden geldigini gormek icin.
  4. Kayitlari data/canli/<katilimci>/ altina egitimdeki .npz bicimiyle yazar (ileride ince ayar icin).

Ornek:
    python ml/live_analysis.py runs/gonullu_testi --models gru226,cnn_transformer_all_s0
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from features import ROOT, el_gorunen_kisim, fix_length, load_sign_names, normalize, to_vec
from models import load_checkpoint


def kayit_oku(path):
    """Telefon JSON'u -> extract_keypoints.py ciktisiyla ayni sozluk (koordinatlar piksel biriminde)."""
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    w, h = r["w"], r["h"]
    t = len(r["kareler"])
    hands = np.zeros((t, 2, 21, 3), np.float32)
    pose = np.zeros((t, 7, 3), np.float32)
    hok = np.zeros((t, 2), bool)
    pok = np.zeros(t, bool)
    for i, k in enumerate(r["kareler"]):
        for s, el in enumerate(k["el"]):
            if el is not None:
                a = np.array(el, np.float32).reshape(21, 2)
                hands[i, s, :, 0], hands[i, s, :, 1] = a[:, 0] * w, a[:, 1] * h
                hok[i, s] = True
        if k["govde"] is not None:
            a = np.array(k["govde"], np.float32).reshape(7, 2)
            pose[i, :, 0], pose[i, :, 1] = a[:, 0] * w, a[:, 1] * h
            pok[i] = True
    return r, {"hands": hands, "pose": pose, "hand_ok": hok, "pose_ok": pok}


def ozellikler(d, genislik):
    """Bir kaydi ozetleyen sayilar (AUTSL ile karsilastirma icin)."""
    hands, pose, hok = normalize(d)
    sh = np.linalg.norm(d["pose"][:, 1, :2] - d["pose"][:, 2, :2], axis=1)
    sh = sh[d["pose_ok"]]
    bilek = hands[:, :, 0, :]  # [T, 2, 2]
    hiz = np.linalg.norm(np.diff(bilek, axis=0), axis=-1)[(hok[1:] * hok[:-1]) > 0]
    p = d["pose"][d["pose_ok"], :, :2]
    bilek_hizi = np.linalg.norm(np.diff(p[:, 5:7], axis=0), axis=2).mean(0) if len(p) > 2 else np.array([np.nan, np.nan])
    return {
        "sol_baskin": float(bilek_hizi[0] > bilek_hizi[1]) if np.isfinite(bilek_hizi).all() else np.nan,
        "kare": len(hands),
        "el_orani": float(hok.mean()),
        "iki_el": float((hok.sum(1) == 2).mean()),
        "uzaklik": float(np.median(sh) / genislik) if len(sh) else np.nan,  # omuz genisligi / goruntu genisligi
        "bilek_y": float(np.median(bilek[..., 1][hok > 0])) if hok.any() else np.nan,
        "hiz": float(np.median(hiz)) if len(hiz) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor", nargs="?", default=str(ROOT / "runs" / "gonullu_testi"))
    ap.add_argument("--models", default="gru226,cnn_transformer_all_s0")
    ap.add_argument("--autsl", type=int, default=400, help="karsilastirma icin AUTSL test orneklerinden kac tane")
    ap.add_argument("--npz", default=str(ROOT / "data" / "canli"))
    a = ap.parse_args()

    dosyalar = sorted(Path(a.klasor).rglob("*.json"))
    if not dosyalar:
        raise SystemExit(f"{a.klasor} icinde kayit yok. Once telefondan alin (dosyanin basindaki adb komutu).")
    names = load_sign_names(ROOT / "docs" / "SignList_ClassId_TR_EN.csv")

    kayitlar = []
    for f in dosyalar:
        r, d = kayit_oku(f)
        d = el_gorunen_kisim(d, pay=1, en_az=8) or d  # uygulamadaki gibi: elin ilk-son goruldugu kareye kirp
        kayitlar.append((f, r, d))
        out = Path(a.npz) / r.get("katilimci", "anonim")
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{f.stem}__{r['hedef_id']}.npz", **d)
    print(f"{len(kayitlar)} kayit, {len({r.get('katilimci') for _, r, _ in kayitlar})} katilimci "
          f"(npz kopyalari: {a.npz})\n")

    y = np.array([r["hedef_id"] for _, r, _ in kayitlar])
    vecs = np.stack([to_vec(*fix_length(*normalize(d))) for _, _, d in kayitlar])
    tel = np.array([r["tahmin"][0]["id"] for _, r, _ in kayitlar])
    tel_p = np.array([r["tahmin"][0]["p"] for _, r, _ in kayitlar])
    esik = kayitlar[0][1].get("esik", 0.6)

    def satir(ad, top1, top3, p):
        kabul = p >= esik
        return (f"{ad:<28} Top-1 %{top1.mean() * 100:5.1f}   Top-3 %{top3.mean() * 100:5.1f}   "
                f"esigi gecen %{kabul.mean() * 100:5.1f}, bunlarda dogruluk "
                f"%{(top1[kabul].mean() * 100 if kabul.any() else float('nan')):5.1f}")

    tel3 = np.array([any(t["id"] == h for t in r["tahmin"]) for (_, r, _), h in zip(kayitlar, y)])
    print(satir("telefon (kayitli tahmin)", tel == y, tel3, tel_p))
    for name in [m for m in a.models.split(",") if m]:
        model, ck = load_checkpoint(name)
        cls = ck["classes"]
        with torch.no_grad():
            prob = model(torch.tensor(vecs)).softmax(1).numpy()
        sira = np.argsort(-prob, 1)[:, :3]
        sinif = np.array(cls)[sira]
        print(satir(name, sinif[:, 0] == y, (sinif == y[:, None]).any(1), prob.max(1)))
        if name == "gru226":
            print(f"{'':<28} telefonla ayni tahmin: %{(sinif[:, 0] == tel).mean() * 100:.1f}")

    print("\nKatilimci bazinda (telefon tahmini):")
    grup = defaultdict(list)
    for (_, r, _), t, h in zip(kayitlar, tel, y):
        grup[r.get("katilimci", "anonim")].append(t == h)
    for k, v in sorted(grup.items()):
        print(f"  {k:<10} %{np.mean(v) * 100:5.1f}  ({sum(v)} / {len(v)})")

    yanlis = [(names.get(h, h), names.get(t, t)) for t, h in zip(tel, y) if t != h]
    if yanlis:
        print("\nYanlislar (hedef -> tahmin):", ", ".join(f"{h}->{t}" for h, t in yanlis[:30]))

    # AUTSL ile karsilastirma
    canli = [ozellikler(d, r["w"]) for _, r, d in kayitlar]
    rng = np.random.default_rng(0)
    autsl = []
    labels = list(csv.reader(open(ROOT / "data" / "archive (1)" / "test_labels.csv")))
    for name, _ in [labels[i] for i in rng.choice(len(labels), min(a.autsl, len(labels)), replace=False)]:
        f = ROOT / "data" / "keypoints" / "test" / f"{name}.npz"
        if f.exists():
            autsl.append(ozellikler(dict(np.load(f)), 1.0))  # AUTSL noktalari 0-1 biriminde (512x512)
    print(f"\nCanli kayitlar ile AUTSL testi (medyan, sol_baskin icin oran; AUTSL {len(autsl)} ornek):")
    aciklama = {
        "sol_baskin": "govdenin 'sol' bilegi daha hizli olan kayit orani (aynali goruntude yuksek, aynasizda dusuk cikar)",
        "kare": "kare sayisi (AUTSL 30 FPS, telefon ~9-12 FPS)",
        "el_orani": "elin gorundugu kare orani",
        "iki_el": "iki elin birlikte gorundugu kare orani",
        "uzaklik": "omuz genisligi / goruntu genisligi (buyuk = kameraya yakin)",
        "bilek_y": "bilek yuksekligi (omuz ortasina gore, omuz genisligi biriminde; + = asagi)",
        "hiz": "kareler arasi bilek hareketi (omuz genisligi biriminde)",
    }
    for k, acik in aciklama.items():
        ozet = np.nanmean if k == "sol_baskin" else np.nanmedian
        c = ozet([o[k] for o in canli])
        u = ozet([o[k] for o in autsl])
        print(f"  {k:<9} canli {c:7.3f}   AUTSL {u:7.3f}   {acik}")


if __name__ == "__main__":
    main()
