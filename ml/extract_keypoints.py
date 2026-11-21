"""AUTSL videolarindan MediaPipe Tasks API ile el + ust govde noktalarini cikarir.

Cikti (video basina bir .npz):
  hands : float32 [T, 2, 21, 3]  slot 0 = "Left" etiketli el, slot 1 = "Right" (MediaPipe etiketi, aynalama notu icin PLAN.md P-07)
  pose  : float32 [T, 7, 3]      burun, sol/sag omuz, sol/sag dirsek, sol/sag bilek
  hand_ok : bool [T, 2]          o karede el bulundu mu
  pose_ok : bool [T]
Ham (normalize edilmemis) goruntu-uzayi koordinatlaridir; normalizasyon egitim tarafinda yapilir.
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]  # BlazePose: burun, omuzlar, dirsekler, bilekler


def make_landmarkers():
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_lite.task")),
            running_mode=vision.RunningMode.VIDEO,
        )
    )
    return hand, pose


def extract_video(path, step=1):
    """step > 1: yalnizca her step. kare islenir (ASL Citizen on egitimi icin 15 FPS); AUTSL'de 1."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"acilamadi: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hand_lm, pose_lm = make_landmarkers()
    hands, pose, hand_ok, pose_ok = [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step:
            i += 1
            continue
        ts = int(i * 1000.0 / fps)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h = np.zeros((2, 21, 3), np.float32)
        hok = np.zeros(2, bool)
        res = hand_lm.detect_for_video(img, ts)
        for lms, handed in zip(res.hand_landmarks, res.handedness):
            slot = 0 if handed[0].category_name == "Left" else 1
            h[slot] = [[p.x, p.y, p.z] for p in lms]
            hok[slot] = True
        p = np.zeros((7, 3), np.float32)
        pres = pose_lm.detect_for_video(img, ts)
        pok = bool(pres.pose_landmarks)
        if pok:
            lms = pres.pose_landmarks[0]
            p = np.array([[lms[j].x, lms[j].y, lms[j].z] for j in POSE_IDX], np.float32)
        hands.append(h); hand_ok.append(hok); pose.append(p); pose_ok.append(pok)
        i += 1
    cap.release()
    hand_lm.close(); pose_lm.close()
    return dict(
        hands=np.stack(hands), pose=np.stack(pose),
        hand_ok=np.stack(hand_ok), pose_ok=np.array(pose_ok),
    )


def work(args):
    src, dst = args
    if dst.exists():
        return str(src), "skip"
    try:
        np.savez_compressed(dst, **extract_video(src))
        return str(src), "ok"
    except Exception as e:  # bozuk video sayilsin, durma
        return str(src), f"HATA {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE.parent / "data" / "archive (1)"))
    ap.add_argument("--out", default=str(HERE.parent / "data" / "keypoints"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--classes", default="", help="virgullu sinif no listesi; bos = hepsi")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="k/n: videolari n parcaya bol, k. parcayi isle (0 tabanli); birden cok makinede paralel calismak icin")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    keep = {int(c) for c in a.classes.split(",") if c} or None
    rows = [r for r in csv.reader(open(Path(a.data) / f"{a.split}_labels.csv"))]
    if keep:
        rows = [r for r in rows if int(r[1]) in keep]
    k, n = (int(x) for x in a.shard.split("/"))
    rows = rows[k::n]
    if a.limit:
        rows = rows[: a.limit]
    out = Path(a.out) / a.split
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(Path(a.data) / a.split / f"{r[0]}_color.mp4", out / f"{r[0]}.npz") for r in rows]
    print(f"{a.split}: {len(jobs)} video, {a.workers} isci", flush=True)

    bad = 0
    with Pool(a.workers) as pool:
        for n, (src, st) in enumerate(pool.imap_unordered(work, jobs), 1):
            if st.startswith("HATA"):
                bad += 1
                print(src, st, file=sys.stderr, flush=True)
            if n % 50 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} (hata: {bad})", flush=True)


if __name__ == "__main__":
    main()
