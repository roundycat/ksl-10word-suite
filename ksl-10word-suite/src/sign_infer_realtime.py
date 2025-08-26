# -*- coding: utf-8 -*-
import cv2
import time
import math
import threading
from collections import deque
import numpy as np

# ---- 설정 불러오기(없으면 기본값 사용) ----
try:
    from config import (
        DEBUG_DRAW, USE_ROI_FOR_HANDS,
        WINDOW, TRIGGER_COUNT, COOLDOWN_SEC, GIYUK_THRESHOLDS,
    )
except Exception:
    DEBUG_DRAW = True
    USE_ROI_FOR_HANDS = True
    WINDOW = 15
    TRIGGER_COUNT = 9
    COOLDOWN_SEC = 1.2
    GIYUK_THRESHOLDS = dict(
        ANGLE_MIN=60.0, ANGLE_MAX=120.0,
        IDX_EXTENDED_MIN=0.35, MID_EXTENDED_MAX=0.33,
        NEAR_FACE_MAX=1.2,
    )

ANGLE_MIN        = GIYUK_THRESHOLDS["ANGLE_MIN"]
ANGLE_MAX        = GIYUK_THRESHOLDS["ANGLE_MAX"]
IDX_EXTENDED_MIN = GIYUK_THRESHOLDS["IDX_EXTENDED_MIN"]
MID_EXTENDED_MAX = GIYUK_THRESHOLDS["MID_EXTENDED_MAX"]
NEAR_FACE_MAX    = GIYUK_THRESHOLDS["NEAR_FACE_MAX"]

# ---- TTS (Windows SAPI) ----
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    for v in _tts_engine.getProperty('voices'):
        nid = (getattr(v, "id", "") or "").lower()
        nname = (getattr(v, "name", "") or "").lower()
        if "korean" in nname or "ko_" in nid or "ko-" in nid:
            _tts_engine.setProperty('voice', v.id)
            break
    _tts_engine.setProperty('rate', 170)
    _tts_engine.setProperty('volume', 1.0)
except Exception as e:
    _tts_engine = None
    print("[TTS] pyttsx3 초기화 실패:", e)

def speak_async(text: str):
    if _tts_engine is None:
        print("[TTS대신표시] ->", text); return
    def _run():
        try:
            _tts_engine.say(text)
            _tts_engine.runAndWait()
        except Exception as e:
            print("[TTS 오류]", e)
    threading.Thread(target=_run, daemon=True).start()

# ---- 유틸 ----
def angle_between(v1, v2, eps=1e-6):
    a = np.linalg.norm(v1) + eps
    b = np.linalg.norm(v2) + eps
    cosang = np.clip(np.dot(v1, v2) / (a * b), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def normalize(pt, nose, eye_dist):
    return (pt - nose) / (eye_dist + 1e-6)

def face_keypoints_to_pixels(det, w, h, mp_fd):
    kps = det.location_data.relative_keypoints
    def rp(idx):
        return np.array([kps[idx].x * w, kps[idx].y * h], dtype=np.float32)
    pts = {
        "right_eye": rp(mp_fd.FaceKeyPoint.RIGHT_EYE),
        "left_eye":  rp(mp_fd.FaceKeyPoint.LEFT_EYE),
        "nose":      rp(mp_fd.FaceKeyPoint.NOSE_TIP),
        "mouth":     rp(mp_fd.FaceKeyPoint.MOUTH_CENTER),
        "right_ear": rp(mp_fd.FaceKeyPoint.RIGHT_EAR_TRAGION),
        "left_ear":  rp(mp_fd.FaceKeyPoint.LEFT_EAR_TRAGION),
    }
    rel = det.location_data.relative_bounding_box
    bbox = (int(rel.xmin*w), int(rel.ymin*h), int(rel.width*w), int(rel.height*h))
    return pts, bbox

def expand_roi(bbox, w, h, sx=1.8, sy=2.2):
    x, y, bw, bh = bbox
    cx, cy = x + bw/2, y + bh/2
    nw, nh = int(bw*sx), int(bh*sy)
    nx, ny = int(cx - nw/2), int(cy - nh/2)
    nx = max(0, nx); ny = max(0, ny)
    nx2 = min(w, nx + nw); ny2 = min(h, ny + nh)
    return (nx, ny, nx2 - nx, ny2 - ny)

def hand_lm_to_full(landmarks, roi):
    x0, y0, rw, rh = roi
    pts = []
    for lm in landmarks:
        px = x0 + lm.x * rw
        py = y0 + lm.y * rh
        pz = lm.z
        pts.append(np.array([px, py, pz], dtype=np.float32))
    return pts

def open_any_camera(indices=(0,1,2,3,4)):
    for i in indices:
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[카메라] 사용 장치 인덱스: {i}")
                return cap
        cap.release()
    raise RuntimeError("웹캠을 찾을 수 없습니다. 연결/권한을 확인하세요.")

def is_giyuk_hand(nw, nidx, nth, nmid, nmouth):
    v1 = nidx - nw
    v2 = nth  - nw
    ang = angle_between(v1[:2], v2[:2])
    idx_len = np.linalg.norm(v1[:2])
    mid_len = np.linalg.norm((nmid - nw)[:2])
    near = np.linalg.norm((nidx - nmouth)[:2])

    cond_ang  = (ANGLE_MIN <= ang <= ANGLE_MAX)
    cond_idx  = (idx_len >= IDX_EXTENDED_MIN)
    cond_mid  = (mid_len <= MID_EXTENDED_MAX)
    cond_near = (near <= NEAR_FACE_MAX)
    ok = cond_ang and cond_idx and cond_mid and cond_near
    return ok, {"ang":ang, "idx_len":idx_len, "mid_len":mid_len, "near":near}

# ---- 메인 ----
def main():
    import mediapipe as mp
    mp_fd = mp.solutions.face_detection
    mp_hands = mp.solutions.hands

    face = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    )
    drawing = mp.solutions.drawing_utils

    cap = open_any_camera()

    giyuk_window = deque(maxlen=WINDOW)
    last_say_time = 0.0

    print("[사용법] q: 종료 / d: 디버그 토글 / r: ROI 토글")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임 캡처 실패"); break

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1) 얼굴 탐지
        face_results = face.process(frame_rgb)
        if not face_results.detections:
            if DEBUG_DRAW:
                cv2.putText(frame, "No face", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow("KSL ㄱ + TTS", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        det = max(face_results.detections,
                  key=lambda d: d.location_data.relative_bounding_box.width *
                                d.location_data.relative_bounding_box.height)
        face_pts, face_bbox = face_keypoints_to_pixels(det, w, h, mp_fd)
        nose  = face_pts["nose"]
        le, re = face_pts["left_eye"], face_pts["right_eye"]
        eye_dist = np.linalg.norm(le - re) + 1e-6
        mouth = face_pts["mouth"]

        # 2) ROI
        roi = (0,0,w,h)
        roi_img = frame_rgb
        if USE_ROI_FOR_HANDS and face_bbox is not None:
            roi = expand_roi(face_bbox, w, h, sx=1.8, sy=2.2)
            x0, y0, rw, rh = roi
            roi_img = frame_rgb[y0:y0+rh, x0:x0+rw].copy()

        # 3) 손 랜드마크
        hands_results = hands.process(roi_img)
        detected_this_frame = False

        if hands_results.multi_hand_landmarks:
            for hand_lms, handedness in zip(hands_results.multi_hand_landmarks,
                                            hands_results.multi_handedness):
                pts3 = hand_lm_to_full(hand_lms.landmark, roi)
                wrist = np.array(pts3[0][:2])
                thumb_tip = np.array(pts3[4][:2])
                index_tip = np.array(pts3[8][:2])
                middle_tip = np.array(pts3[12][:2])

                nw   = normalize(np.append(wrist, 0), np.append(nose, 0), np.float32(eye_dist))
                nth  = normalize(np.append(thumb_tip, 0), np.append(nose, 0), np.float32(eye_dist))
                nidx = normalize(np.append(index_tip, 0), np.append(nose, 0), np.float32(eye_dist))
                nmid = normalize(np.append(middle_tip, 0), np.append(nose, 0), np.float32(eye_dist))
                nmouth = normalize(np.append(mouth, 0), np.append(nose, 0), np.float32(eye_dist))

                ok_g, stats = is_giyuk_hand(nw, nidx, nth, nmid, nmouth)
                detected_this_frame |= ok_g

                if DEBUG_DRAW:
                    drawing.draw_landmarks(
                        frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2),
                        connection_drawing_spec=drawing.DrawingSpec(color=(0,200,0), thickness=1)
                    )
                    if USE_ROI_FOR_HANDS:
                        x0, y0, rw, rh = roi
                        cv2.rectangle(frame, (x0, y0), (x0+rw, y0+rh), (255,200,0), 1)
                    cx, cy = int(index_tip[0]), int(index_tip[1])
                    lab = handedness.classification[0].label
                    cv2.putText(
                        frame,
                        f"{lab} | ang:{stats['ang']:.0f} idx:{stats['idx_len']:.2f} mid:{stats['mid_len']:.2f} near:{stats['near']:.2f}",
                        (max(10, cx-120), max(20, cy-20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255 if ok_g else 200, 0 if ok_g else 200), 1, cv2.LINE_AA
                    )

        # 4) 윈도우 투표
        giyuk_window.append(1 if detected_this_frame else 0)
        votes = sum(giyuk_window)

        # 5) TTS 트리거
        now = time.time()
        if votes >= TRIGGER_COUNT and now - last_say_time >= COOLDOWN_SEC:
            last_say_time = now
            speak_async("기역")
            if DEBUG_DRAW:
                cv2.putText(frame, "Detected: ㄱ (TTS)", (20, h-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 3, cv2.LINE_AA)
            giyuk_window.clear()

        # 6) 디버그: 얼굴 표시/투표
        if DEBUG_DRAW:
            for k in ["left_eye","right_eye","nose","mouth"]:
                p = face_pts[k].astype(int)
                cv2.circle(frame, tuple(p), 3, (0,128,255), -1)
            x, y, bw, bh = face_bbox
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0,128,255), 1)
            cv2.putText(frame, f"Votes:{votes}/{WINDOW}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)

        cv2.imshow("KSL ㄱ + TTS", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            globals()['DEBUG_DRAW'] = not globals()['DEBUG_DRAW']
        elif key == ord('r'):
            globals()['USE_ROI_FOR_HANDS'] = not globals()['USE_ROI_FOR_HANDS']

    cap.release()
    cv2.destroyAllWindows()
    face.close()
    hands.close()

if __name__ == "__main__":
    main()
