"""ASL Citizen videolarindan (Amerikan Isaret Dili) el + govde noktalarini cikarir; on egitim icin.

Veri seti 42,8 GB'lik tek bir zip. Tamamini indirmek yerine zip uzaktan okunur: her isci kendi videosunu
HTTP aralik istegiyle ceker, gecici dosyaya yazar, AUTSL'deki ayni fonksiyonla (extract_keypoints.py)
noktalari cikarir ve videoyu siler. Diskte yalnizca .npz ciktilari kalir (~2-3 GB). Yarida kesilirse
var olan ciktilari atlayarak devam eder.

Lisans: Microsoft Research License (yalnizca ticari olmayan arastirma). Ayrinti: data/asl_citizen/use.txt

Ornek:
    python ml/extract_asl_citizen.py --workers 14
    python ml/extract_asl_citizen.py --splits val --limit 24     # hiz denemesi
"""
import argparse
import csv
import io
import os
import sys
import tempfile
import time
import urllib.request
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from extract_keypoints import extract_video

ROOT = Path(__file__).resolve().parent.parent
URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"


class UzakDosya:
    """zipfile'in okuyabilecegi, HTTP aralik istekleriyle calisan salt-okunur dosya."""

    def __init__(self, url):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.n = int(r.headers["Content-Length"])
        self.p = 0

    def seekable(self):
        return True

    def tell(self):
        return self.p

    def seek(self, o, nereden=0):
        self.p = o if nereden == 0 else (self.p + o if nereden == 1 else self.n + o)
        return self.p

    def read(self, k=-1):
        k = self.n - self.p if k < 0 else min(k, self.n - self.p)
        if k <= 0:
            return b""
        for deneme in range(8):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.p}-{self.p + k - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    veri = r.read()
                if len(veri) == k:
                    self.p += k
                    return veri
            except Exception:
                pass
            time.sleep(3 * (deneme + 1))
        raise IOError(f"okunamadi: {self.p}+{k}")


_zip = None
ADIM = 2  # her 2. kare (15 FPS): hiz icin; telefon zaten ~10 FPS'te calisiyor


def _baslat():
    global _zip
    _zip = zipfile.ZipFile(UzakDosya(URL))


def isle(is_):
    uye, hedef = is_
    if hedef.exists():
        return uye, "atla"
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
            t.write(_zip.read(uye))
            gecici = t.name
        try:
            np.savez_compressed(hedef, **extract_video(gecici, step=ADIM))
        finally:
            os.unlink(gecici)
        return uye, "tamam"
    except Exception as e:
        return uye, f"HATA {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "asl_citizen"))
    ap.add_argument("--splits", default="train,val", help="ASL'nin kendi test bolmesi on egitimde gerekmiyor")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="her bolmeden en fazla kac video (deneme icin)")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(UzakDosya(URL))
    for ad in ("splits/train.csv", "splits/val.csv", "splits/test.csv", "use.txt"):
        hedef = out / ad
        if not hedef.exists():
            hedef.write_bytes(z.read(f"ASL_Citizen/{ad}"))

    isler = []
    for bolum in a.splits.split(","):
        (out / "keypoints" / bolum).mkdir(parents=True, exist_ok=True)
        satirlar = list(csv.DictReader(open(out / "splits" / f"{bolum}.csv", encoding="utf-8")))
        if a.limit:
            satirlar = satirlar[: a.limit]
        for r in satirlar:
            dosya = r["Video file"]
            isler.append((f"ASL_Citizen/videos/{dosya}", out / "keypoints" / bolum / (Path(dosya).stem + ".npz")))
    kalan = [i for i in isler if not i[1].exists()]
    print(f"{len(isler)} video, {len(isler) - len(kalan)} tanesi zaten var, {a.workers} isci", flush=True)

    t0, hata = time.time(), 0
    with Pool(a.workers, initializer=_baslat) as havuz:
        for k, (uye, durum) in enumerate(havuz.imap_unordered(isle, kalan), 1):
            if durum.startswith("HATA"):
                hata += 1
                print(uye, durum, file=sys.stderr, flush=True)
            if k % 100 == 0 or k == len(kalan):
                hiz = k / (time.time() - t0)
                print(f"{k}/{len(kalan)} | {hiz:.2f} video/sn | kalan ~{(len(kalan) - k) / hiz / 3600:.1f} sa | hata {hata}",
                      flush=True)


if __name__ == "__main__":
    main()
