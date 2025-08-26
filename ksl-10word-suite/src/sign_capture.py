# -*- coding: utf-8 -*-
import cv2, json, time
import numpy as np
import mediapipe as mp
from pathlib import Path
from datetime import datetime
from config import CAM_INDEX, FRAME_SIZE

mp_hands = mp.solutions.hands

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def capture_segment(label="안녕하세요", user="u01", out_dir="data/signs/jsonl"):
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    hands = mp_hands.Hands(False, 2, 0.6, 0.6)
    print("Space: 시작/정지  |  q: 종료  |  label:", label)

    recording=False
    seq=[]

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        h,w = frame.shape[:2]
        left=None; right=None
        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, handeness in zip(res.multi_hand_landmarks, res.multi_handedness):
                label_h = handeness.classification[0].label
                xy = [[p.x*w, p.y*h] for p in lm.landmark]
                if label_h == "Left": left = xy
                else: right = xy
            for lm in res.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        if recording:
            seq.append({"left": left, "right": right})

        cv2.putText(frame, f"LABEL:{label}  REC:{'ON' if recording else 'OFF'}  frames:{len(seq)}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0) if recording else (0,200,255),2)
        cv2.putText(frame, "Space:start/stop  q:quit", (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.7,(50,200,255),2)
        cv2.imshow("Sign Capture", frame)
        k = cv2.waitKey(1) & 0xFF
        if k==ord(' '):
            recording = not recording
            if not recording and len(seq)>0:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                out_path = Path(out_dir)/f"{label}.jsonl"
                with open(out_path,"a",encoding="utf-8") as f:
                    rec = {"label":label,"user":user,"fps":30,"frames":seq,"ts":ts}
                    f.write(json.dumps(rec, ensure_ascii=False)+"\n")
                print("saved:", out_path, "len:", len(seq))
                seq=[]
        elif k==ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="안녕하세요")
    ap.add_argument("--user", default="u01")
    ap.add_argument("--out_dir", default="data/signs/jsonl")
    args = ap.parse_args()
    capture_segment(args.label, args.user, args.out_dir)
