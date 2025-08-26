# 

# -*- coding: utf-8 -*-
"""
sign_train_dtw.py
- data/signs/jsonl/*.jsonl  (각 라인: {"label","frames":[{"left","right","face"}, ...]})
- 얼굴+손 랜드마크로 feature seq 생성 → 증강 → 프로토타입(k개) → 라벨별 thr/margin 계산
- 저장: models/sign_dtw.pkl
"""
import os, json, math, random, argparse
from pathlib import Path

import numpy as np
import joblib
import re

from sign_features import frame_feature, resample_seq, dtw_distance

# (선택) 특정 라벨만 학습하고 싶으면 config.SIGN_LABELS 사용
try:
    from config import SIGN_LABELS
except Exception:
    SIGN_LABELS = None  # 전체 라벨 학습

# ----------------- 유틸 -----------------
def _np(x):
    return None if x is None else np.asarray(x, dtype=np.float32)

def load_raw_sequences(data_dir: str):
    """
    JSONL들을 라벨 -> [원본프레임열(list[dict]), ...] 형태로 로드.
    - BOM(utf-8-sig) 안전
    - 빈 줄 무시
    - 잘못된 백슬래시 이스케이프 자동 보정 (\ -> \\), 단 정상 이스케이프는 유지
    """
    data = {}
    p = Path(data_dir)
    if not p.exists():
        print(f"[WARN] data_dir not found: {data_dir}")
        return data

    # 유효 이스케이프가 아닌 백슬래시만 잡아서 \\ 로 치환하는 정규식
    esc_fix = re.compile(r'\\(?!["\\/bfnrtu])')

    for f in sorted(p.glob("*.jsonl")):
        lab = f.stem
        if "SIGN_LABELS" in globals() and SIGN_LABELS and (lab not in SIGN_LABELS):
            continue

        try:
            with open(f, "r", encoding="utf-8-sig") as fh:  # BOM 안전
                for ln, line in enumerate(fh, 1):
                    s = line.strip().lstrip("\ufeff")
                    if not s:
                        continue
                    try:
                        rec = json.loads(s)
                        if isinstance(rec.get("video"), str):
                            rec["video"] = rec["video"].replace("\\", "/")
                    except json.JSONDecodeError:
                        # 잘못된 \ 만 \\ 로 바꾼 뒤 한 번 더 시도
                        s2 = esc_fix.sub(r'\\\\', s)
                        try:
                            rec = json.loads(s2)
                        except json.JSONDecodeError as e2:
                            print(f"[WARN] JSONL parse fail: {f} line {ln}: {e2}")
                            continue

                    frames = rec.get("frames", [])
                    if frames:
                        data.setdefault(lab, []).append(frames)
        except Exception as e:
            print(f"[WARN] open/read fail: {f}: {e}")
            continue

    return data

# ----------------- 증강 (원본 좌표 기준) -----------------
def aug_rotate_around_nose(frames, max_deg=6.0):
    ang = math.radians(random.uniform(-max_deg, max_deg))
    c, s = math.cos(ang), math.sin(ang)
    out = []
    for fr in frames:
        left = _np(fr.get("left")); right = _np(fr.get("right")); face = fr.get("face")
        if face and ("nose" in face):
            nose = np.asarray(face["nose"], np.float32)
            R = np.array([[c, -s], [s, c]], np.float32)
            def rot(xy):
                if xy is None: return None
                return (xy - nose) @ R.T + nose
            left, right = rot(left), rot(right)
        out.append({
            "left": None if left is None else left.tolist(),
            "right": None if right is None else right.tolist(),
            "face": face
        })
    return out

def aug_add_noise(frames, sigma_px=1.5):
    out = []
    for fr in frames:
        left = _np(fr.get("left")); right = _np(fr.get("right")); face = fr.get("face")
        def j(xy):
            if xy is None: return None
            return (xy + np.random.normal(0, sigma_px, size=xy.shape).astype(np.float32))
        out.append({
            "left": None if left is None else j(left).tolist(),
            "right": None if right is None else j(right).tolist(),
            "face": face
        })
    return out

def aug_time_warp(frames, lo=0.9, hi=1.12):
    L = len(frames)
    r = random.uniform(lo, hi)
    newL = max(4, int(round(L * r)))
    xs = np.linspace(0, L - 1, newL)
    i0 = np.floor(xs).astype(int); i1 = np.minimum(i0 + 1, L - 1)
    frac = xs - i0
    out = []
    for t in range(newL):
        a = frames[int(i0[t])]; b = frames[int(i1[t])]
        fr = {}
        for k in ["left", "right"]:
            pa, pb = _np(a.get(k)), _np(b.get(k))
            if pa is None and pb is None: fr[k] = None
            elif pa is None: fr[k] = pb.tolist()
            elif pb is None: fr[k] = pa.tolist()
            else: fr[k] = (pa * (1 - frac[t]) + pb * frac[t]).tolist()
        fr["face"] = a.get("face") if a.get("face") is not None else b.get("face")
        out.append(fr)
    return out

def aug_flip_lr(frames, enable=True):
    if not enable: return frames
    out = []
    for fr in frames:
        face = fr.get("face")
        if face and ("nose" in face):
            nose = np.asarray(face["nose"], np.float32)
            def flip(xy):
                if xy is None: return None
                xy = np.asarray(xy, np.float32).copy()
                xy[:, 0] = 2 * nose[0] - xy[:, 0]
                return xy
            left = flip(fr.get("left")) if fr.get("left") is not None else None
            right = flip(fr.get("right")) if fr.get("right") is not None else None
            face2 = {k: [float(2 * nose[0] - v[0]), float(v[1])]
                     for k, v in face.items() if isinstance(v, (list, tuple))}
            out.append({
                "left": None if left is None else left.tolist(),
                "right": None if right is None else right.tolist(),
                "face": {**face, **face2}
            })
        else:
            out.append(fr)
    return out

def make_augmented(frames, mult=2, flip_lr=False):
    seqs = [frames]
    for _ in range(mult):
        seqs.append(aug_time_warp(frames))
        seqs.append(aug_rotate_around_nose(frames))
        seqs.append(aug_add_noise(frames))
    if flip_lr:
        seqs.append(aug_flip_lr(frames, True))
    return seqs

# ----------------- Feature / Prototype -----------------
def to_feature_seq(frames):
    Fs = []
    for fr in frames:
        left = _np(fr.get("left")); right = _np(fr.get("right")); face = fr.get("face")
        Fs.append(frame_feature(left, right, face=face))
    return np.stack(Fs, axis=0)  # (Tvar, D)

def make_prototypes(data, T=32, k=3):
    protos = {}
    for lab, seqs in data.items():
        R = [resample_seq(s, T) for s in seqs if len(s) > 0]
        if not R: continue
        M = np.mean(np.stack(R), axis=0)
        dists = [np.mean(np.linalg.norm(r - M, axis=1)) for r in R]
        idx = np.argsort(dists)[:min(k, len(R))]
        protos[lab] = [R[i] for i in idx]
    return protos

def compute_thresholds(protos, data, T=32, band_ratio=0.15):
    """라벨별 수용 임계(thr)와 여유(margin) 계산"""
    thr_map, margin_map = {}, {}
    band = int(round(band_ratio * T))
    for lab, seqs in data.items():
        own_dists, margins = [], []
        for s in seqs:
            r = resample_seq(s, T)
            per_label = []
            for L, plist in protos.items():
                dmin = min(dtw_distance(r, p, band=band) for p in plist)
                per_label.append((L, dmin))
            per_label.sort(key=lambda x: x[1])
            (best_lab, best_d) = per_label[0]
            second_d = per_label[1][1] if len(per_label) > 1 else 1e18
            if best_lab == lab:
                own_dists.append(best_d)
            margins.append(second_d - best_d)
        if own_dists:
            q25, q75 = np.percentile(own_dists, [25, 75])
            iqr = max(1e-6, q75 - q25)
            thr_map[lab] = float(q75 + 1.5 * iqr)
        else:
            thr_map[lab] = float(np.median([d for _, d in per_label]))
        margin_map[lab] = float(np.percentile(margins, 25)) if margins else 0.0
    return thr_map, margin_map

def evaluate_with_reject(protos, data, thr, margin, T=32, band_ratio=0.15):
    band = int(round(band_ratio * T))
    total = 0; ok = 0; rej = 0
    for gt, seqs in data.items():
        for s in seqs:
            total += 1
            r = resample_seq(s, T)
            best = []
            for L, plist in protos.items():
                dmin = min(dtw_distance(r, p, band=band) for p in plist)
                best.append((L, dmin))
            best.sort(key=lambda x: x[1])
            best_lab, best_d = best[0]
            second_d = best[1][1] if len(best) > 1 else 1e18
            if (best_d > float(thr.get(best_lab, best_d + 1.0))) or \
               ((second_d - best_d) < float(margin.get(best_lab, 0.0))):
                rej += 1
                continue
            if best_lab == gt:
                ok += 1
    acc = ok / max(1, (total - rej))
    coverage = (total - rej) / max(1, total)
    return acc, coverage, {"total": total, "accepted": total - rej, "correct": ok}

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/signs/jsonl")
    ap.add_argument("--out", default="models/sign_dtw.pkl")
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--aug_mult", type=int, default=2)
    ap.add_argument("--aug_flip", action="store_true")
    ap.add_argument("--band_ratio", type=float, default=0.15)
    args = ap.parse_args()

    raw = load_raw_sequences(args.data)
    if not raw:
        raise SystemExit("데이터가 없습니다. data/signs/jsonl/*.jsonl 를 확인하세요.")

    # 증강 + feature seq
    data = {}
    for lab, seq_list in raw.items():
        for seq in seq_list:
            for aug_seq in make_augmented(seq, mult=args.aug_mult, flip_lr=args.aug_flip):
                try:
                    F = to_feature_seq(aug_seq)  # (Tvar, D)
                    data.setdefault(lab, []).append(F)
                except Exception:
                    pass

    # 프로토타입 & 임계 계산
    protos = make_prototypes(data, T=args.T, k=args.k)
    thr_map, margin_map = compute_thresholds(protos, data, T=args.T, band_ratio=args.band_ratio)

    # 평가(거절 포함)
    acc, cov, stats = evaluate_with_reject(protos, data, thr_map, margin_map, T=args.T, band_ratio=args.band_ratio)
    print(f"[train-set] acc={acc:.3f}, coverage={cov:.3f}, stats={stats}")

    feat_dim = next(iter(protos.values()))[0].shape[1]
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "protos": protos,
        "T": args.T,
        "feat_dim": int(feat_dim),
        "features": "hand+face_v1",
        "thr": thr_map,
        "margin": margin_map,
        "band_ratio": float(args.band_ratio),
    }, args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
