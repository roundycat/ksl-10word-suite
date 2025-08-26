# -*- coding: utf-8 -*-
import os, csv, json, argparse, cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from config import SIGN_LABELS

def write_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def face_keypoints(det, w, h, mp_fd):
    kps = det.location_data.relative_keypoints
    def rp(idx):
        return [float(kps[idx].x * w), float(kps[idx].y * h)]
    rel = det.location_data.relative_bounding_box
    bbox = [int(rel.xmin*w), int(rel.ymin*h), int(rel.width*w), int(rel.height*h)]
    return {
        "right_eye": rp(mp_fd.FaceKeyPoint.RIGHT_EYE),
        "left_eye":  rp(mp_fd.FaceKeyPoint.LEFT_EYE),
        "nose":      rp(mp_fd.FaceKeyPoint.NOSE_TIP),
        "mouth":     rp(mp_fd.FaceKeyPoint.MOUTH_CENTER),
        "right_ear": rp(mp_fd.FaceKeyPoint.RIGHT_EAR_TRAGION),
        "left_ear":  rp(mp_fd.FaceKeyPoint.LEFT_EAR_TRAGION),
        "bbox": bbox,
    }

def process_segment(cap, start_s, end_s, fps, hands, face0, face1, use_roi=True):
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s*fps))
    total_frames = max(1, int((end_s-start_s)*fps))
    mp_fd = mp.solutions.face_detection

    for _ in range(total_frames):
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 밝기 보정 후 얼굴 탐지(0→1)
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(2.0, (8,8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        bright = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        det_img = cv2.resize(bright, None, fx=0.75, fy=0.75)
        res_face = face0.process(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))
        if not (res_face and res_face.detections):
            res_face = face1.process(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))

        face_pts = None; face_bbox = None
        if res_face and res_face.detections:
            det = max(res_face.detections,
                      key=lambda d: d.location_data.relative_bounding_box.width *
                                    d.location_data.relative_bounding_box.height)
            pts = face_keypoints(det, w, h, mp_fd)
            face_pts = {k:v for k,v in pts.items() if k!="bbox"}
            face_bbox = pts["bbox"]

        # ROI에서 손 탐지(얼굴 있으면 확장 박스, 없으면 전체)
        if use_roi and face_bbox is not None:
            x, y, bw, bh = face_bbox
            cx, cy = x + bw//2, y + bh//2
            rw, rh = int(bw*1.8), int(bh*2.2)
            x0, y0 = max(0, cx-rw//2), max(0, cy-rh//2)
            x1, y1 = min(w, x0+rw), min(h, y0+rh)
            crop = frame[y0:y1, x0:x1]
            res_h = hands.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            left=None; right=None
            if res_h.multi_hand_landmarks and res_h.multi_handedness:
                for lm, handed in zip(res_h.multi_hand_landmarks, res_h.multi_handedness):
                    label_h = handed.classification[0].label
                    xy = [[x0 + p.x * (x1-x0), y0 + p.y * (y1-y0)] for p in lm.landmark]
                    if label_h=="Left": left = xy
                    else: right = xy
        else:
            res_h = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            left=None; right=None
            if res_h.multi_hand_landmarks and res_h.multi_handedness:
                for lm, handed in zip(res_h.multi_hand_landmarks, res_h.multi_handedness):
                    label_h = handed.classification[0].label
                    xy = [[p.x*w, p.y*h] for p in lm.landmark]
                    if label_h=="Left": left = xy
                    else: right = xy

        frames.append({"left": left, "right": right, "face": face_pts})
    return frames

def main(csv_path, out_dir, fps_default=30, use_roi=True):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,          # 0=빠름, 1=기본, 2=정확
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    )
    mp_fd = mp.solutions.face_detection
    face0 = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.3)
    face1 = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.3)

    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            lab = row["label"].strip()
            if SIGN_LABELS and (lab not in SIGN_LABELS):
                print("[SKIP] not in SIGN_LABELS:", lab); continue
            vid = row["video"].strip()
            s = float(row["start"]) if row.get("start","")!="" else 0.0
            e = float(row["end"])   if row.get("end","")!="" else None
            user = row.get("user","u01")
            cap = cv2.VideoCapture(vid)
            fps = cap.get(cv2.CAP_PROP_FPS) or fps_default
            if e is None:
                e = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps*2)/fps
            frames = process_segment(cap, s, e, fps, hands, face0, face1, use_roi=use_roi)
            cap.release()
            rec = {"label":lab,"user":user,"fps":fps,"frames":frames,"video":vid,"range":[s,e]}
            write_jsonl(out_dir/f"{lab}.jsonl", rec)
            print("saved:", lab, vid, f"{s:.2f}-{e:.2f}s", "frames:", len(frames))
    print("done ->", out_dir)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="data/signs/jsonl")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no_roi", action="store_true", help="얼굴 ROI를 사용하지 않음")
    args = ap.parse_args()
    main(args.csv, args.out_dir, args.fps, use_roi=not args.no_roi)
