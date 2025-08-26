# """
# Streamlit: DTW 10단어 실시간 인식 (모드 A)
# - 얼굴+손 상대 피처 + DTW 거절(thr/margin) + ROI + 디버그 + 한글 렌더
# - 인식 실패 연속 시 GPT 도움말(Secrets 기반) - 메인 결과와 분리
# - 느슨 모드(거절 off) / band 비율 조절 / 디버그 HUD
# - ✅ DTW 결과를 GPT로 ‘보강 문장’으로 함께 출력 (옵션)
# - ✅ 보강 문장도 TTS 가능 (옵션)
# - ✅ 출력 형식 선택: 라벨만 / 라벨+디버그 / 디버그만
# - ✅ (fix) 디버그만 모드에서는 REJECT여도 디버그 문자열을 인식 결과에 기록
# - ✅ (추가) 브라우저 TTS 지원: 서버(pyTTSX3) 불가 환경에서도 음성 출력
# """

# import os, sys, time, json, base64, threading
# from pathlib import Path

# import cv2
# import numpy as np
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
# from streamlit.components.v1 import html as _st_html  # ★ 브라우저 TTS용
# import av
# from openai import OpenAI

# # ---------------- Streamlit 페이지 설정 (첫 Streamlit 호출) ----------------
# st.set_page_config(
#     page_title="KSL DTW (모드 A)",
#     layout="wide",
#     page_icon="🤟",
#     initial_sidebar_state="expanded",
# )

# # ---------------- 브라우저 TTS 유틸 ----------------
# def browser_tts(text: str, rate: float = 1.0, pitch: float = 1.0, lang: str = "ko-KR"):
#     """브라우저 음성합성(Web Speech API)로 말하기"""
#     if not text:
#         return
#     safe = json.dumps(text)  # 따옴표/개행 이스케이프
#     js = f"""
#     <script>
#     (function() {{
#         try {{
#             const u = new SpeechSynthesisUtterance({safe});
#             u.lang = "{lang}";
#             u.rate = {rate};
#             u.pitch = {pitch};
#             // 중복 발화 방지: 직전 발화 중단 후 재생
#             window.speechSynthesis.cancel();
#             window.speechSynthesis.speak(u);
#         }} catch(e) {{ console.error(e); }}
#     }})();
#     </script>
#     """
#     _st_html(js, height=0)

# # ---------------- OpenAI (secrets.toml) ----------------
# def _S(name: str):
#     import os, streamlit as st
#     return os.environ.get(name) or st.secrets.get(name) or st.secrets.get("\ufeff"+name)

# def _init_openai_from_secrets():
#     try:
#         from openai import OpenAI
#         key  = _S("OPENAI_API_KEY")
#         base = _S("OPENAI_BASE_URL")
#         org  = _S("OPENAI_ORG")
#         proj = _S("OPENAI_PROJECT")
#         kw = {}
#         if base: kw["base_url"] = base
#         if org:  kw["organization"] = org
#         if proj: kw["project"] = proj
#         return OpenAI(api_key=key, **kw)
#     except Exception as e:
#         print("[OpenAI] secrets 초기화 실패 또는 키 없음:", e)
#         return None

# OPENAI_CLIENT = _init_openai_from_secrets()
# OPENAI_MODEL  = _S("OPENAI_MODEL") or "gpt-4o"

# def _supports_vision(model_name: str) -> bool:
#     n = (model_name or "").lower()
#     return any(k in n for k in ["gpt-5", "gpt-4o", "omni", "4.1"])

# # ---------------- 경로 ----------------
# HERE = Path(__file__).resolve().parent
# ROOT = HERE.parent
# if str(HERE) not in sys.path: sys.path.append(str(HERE))
# if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

# # ---------------- RTC 설정 ----------------
# RTC_CONFIGURATION = {}

# # ---------------- 한글 텍스트 렌더 (Pillow) ----------------
# try:
#     from PIL import Image, ImageDraw, ImageFont
#     def draw_korean_text(bgr, text, xy=(20,30), font_size=28, color=(0,255,0), stroke=2, font_path=None):
#         if font_path is None:
#             for p in [
#                 r"C:\\Windows\\Fonts\\malgun.ttf",
#                 r"C:\\Windows\\Fonts\\malgunbd.ttf",
#                 "/System/Library/Fonts/AppleSDGothicNeo.ttc",
#                 "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
#             ]:
#                 if os.path.exists(p): font_path = p; break
#         if not font_path or not os.path.exists(font_path): return bgr
#         pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
#         draw = ImageDraw.Draw(pil)
#         font = ImageFont.truetype(font_path, font_size)
#         draw.text(xy, text, font=font, fill=(color[2], color[1], color[0]),
#                   stroke_width=stroke, stroke_fill=(0,0,0))
#         return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
# except Exception:
#     def draw_korean_text(bgr, text, xy=(20,30), font_size=28, color=(0,255,0), stroke=2, font_path=None):
#         return bgr

# # ---------------- TTS (서버 pyttsx3) ----------------
# try:
#     import pyttsx3
#     _tts = pyttsx3.init()
#     for v in _tts.getProperty("voices"):
#         nid = (getattr(v, "id", "") or "").lower()
#         nname = (getattr(v, "name", "") or "").lower()
#         if "korean" in nname or "ko_" in nid or "ko-" in nid:
#             _tts.setProperty("voice", v.id); break
#     _tts.setProperty("rate", 170); _tts.setProperty("volume", 1.0)
# except Exception:
#     _tts = None

# def speak_async(text: str):
#     if not text or _tts is None: return
#     def _run():
#         try: _tts.say(text); _tts.runAndWait()
#         except Exception: pass
#     threading.Thread(target=_run, daemon=True).start()

# # ---------------- 모델/템플릿 ----------------
# @st.cache_resource
# def load_dtw_model():
#     import joblib
#     return joblib.load(str(ROOT / "models" / "sign_dtw.pkl"))

# @st.cache_resource
# def load_templates():
#     with open(HERE / "templates_ko.json", "r", encoding="utf-8") as f:
#         return json.load(f)

# # ---------------- 비디오 프로세서 ----------------
# class DTWProcessor(VideoProcessorBase):
#     """DTW + ROI + 얼굴/손 피처 + 리젝트 + GPT 도움말/보강 + 디버그 HUD + 브라우저/서버 TTS"""
#     def __init__(self, model, templates, speak=True, use_roi=True, debug=True,
#                  use_gpt=False, use_gpt_vision=False,
#                  use_reject=True, band_override=None,
#                  use_gpt_enhance=True, gpt_speak=True, gpt_temp=0.2,
#                  debug_out_mode="label"):
#         import mediapipe as mp
#         self.mp = mp
#         self.hands = mp.solutions.hands.Hands(
#             static_image_mode=False, max_num_hands=2,
#             model_complexity=1,
#             min_detection_confidence=0.6, min_tracking_confidence=0.6
#         )
#         self.face0 = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)
#         self.face1 = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
#         self.drawing = mp.solutions.drawing_utils
#         self.mp_fd = mp.solutions.face_detection

#         self.model = model
#         self.templates = templates
#         self.protos = model["protos"]; self.T = int(model["T"])
#         self.band_ratio = float(model.get("band_ratio", 0.15))
#         self.thr_map = model.get("thr", {})
#         self.margin_map = model.get("margin", {})

#         self.speak = speak
#         self.use_roi = use_roi
#         self.debug = debug
#         self.roi_scale = (1.8, 2.2)

#         # 상태
#         self.active = False; self.seq = []
#         self.last_feat = None; self.idle_count = 0; self.act_count = 0
#         self.text_out = ""; self.last_label = ""
#         self.last_display = ""   # 프레임 오버레이 표시 문자열
#         self._cur_face = None
#         self._last_two = []  # 연속 확정용

#         # GPT 폴백 상태
#         self.use_gpt = use_gpt
#         self.use_gpt_vision = use_gpt_vision
#         self._nohit_streak = 0
#         self._gpt_busy = False
#         self._last_gpt_time = 0.0
#         self.gpt_help = ""
#         self.gpt_err = ""

#         # GPT 보강(성공 시 한 문장)
#         self.use_gpt_enhance = bool(use_gpt_enhance)
#         self.gpt_speak = bool(gpt_speak)
#         self.gpt_temp = float(gpt_temp)
#         self.aug_text = ""
#         self._aug_busy = False
#         self._last_aug_time = 0.0

#         # 리젝트/밴드
#         self.use_reject = bool(use_reject)
#         self.band_override = float(band_override) if band_override is not None else None

#         # 출력 모드: "label" | "label+debug" | "debug"
#         self.debug_out_mode = debug_out_mode

#         # 프로토타입 차원
#         any_lab = next(iter(self.protos)); any_proto = self.protos[any_lab][0]
#         self.proto_dim = int(any_proto.shape[1])

#         from sign_features import frame_feature, resample_seq, dtw_distance
#         self.frame_feature = frame_feature
#         self.resample_seq = resample_seq
#         self.dtw_distance = dtw_distance

#         self.debug_info = ""

#         # 🔊 TTS 관련 상태
#         self._last_tts = ""
#         self._last_tts_time = 0.0
#         self._tts_cooldown = 1.8  # 초: 같은 문구 연속 발화 방지
#         self.pending_tts = []     # 브라우저 TTS 큐

#     # --- 유틸 ---
#     def _face_keypoints_to_pixels(self, det, w, h):
#         kps = det.location_data.relative_keypoints
#         def rp(idx):
#             return np.array([kps[idx].x * w, kps[idx].y * h], dtype=np.float32)
#         pts = {
#             "right_eye": rp(self.mp_fd.FaceKeyPoint.RIGHT_EYE),
#             "left_eye":  rp(self.mp_fd.FaceKeyPoint.LEFT_EYE),
#             "nose":      rp(self.mp_fd.FaceKeyPoint.NOSE_TIP),
#             "mouth":     rp(self.mp_fd.FaceKeyPoint.MOUTH_CENTER),
#             "right_ear": rp(self.mp_fd.FaceKeyPoint.RIGHT_EAR_TRAGION),
#             "left_ear":  rp(self.mp_fd.FaceKeyPoint.LEFT_EAR_TRAGION),
#         }
#         rel = det.location_data.relative_bounding_box
#         bbox = (int(rel.xmin*w), int(rel.ymin*h), int(rel.width*w), int(rel.height*h))
#         return pts, bbox

#     def _expand_roi(self, bbox, w, h, sx=1.8, sy=2.2):
#         x, y, bw, bh = bbox
#         cx, cy = x + bw/2, y + bh/2
#         nw, nh = int(bw*sx), int(bh*sy)
#         nx, ny = int(cx - nw/2), int(cy - nh/2)
#         nx = max(0, nx); ny = max(0, ny)
#         nx2 = min(w, nx + nw); ny2 = min(h, ny + nh)
#         return (nx, ny, nx2 - nx, ny2 - ny)

#     # --- 세그먼트 ---
#     def _segment_step(self, left_xy, right_xy):
#         feat = self.frame_feature(left_xy, right_xy, face=self._cur_face)
#         if feat.shape[0] != self.proto_dim:
#             feat = self.frame_feature(left_xy, right_xy, face=None)

#         if self.last_feat is None: self.last_feat = feat
#         mv = float(np.linalg.norm(feat - self.last_feat)); self.last_feat = feat

#         hand_present = (left_xy is not None) or (right_xy is not None)
#         self.act_count = min(10, self.act_count + 1) if (hand_present and mv > 0.10) else max(0, self.act_count - 1)
#         self.idle_count = min(10, self.idle_count + 1) if ((not hand_present) or mv < 0.03) else max(0, self.idle_count - 1)

#         if not self.active and self.act_count >= 7:
#             self.active = True; self.seq = []
#         if self.active:
#             self.seq.append(feat)
#             if self.idle_count >= 7:
#                 self.active = False
#                 return True
#         return False

#     # --- 분류 (+상위 후보/지표) ---
#     def _classify(self, want_stats=False):
#         if not self.seq:
#             return (None, []) if want_stats else None

#         r = self.resample_seq(np.stack(self.seq), self.T)
#         band_ratio = self.band_override if self.band_override is not None else self.band_ratio
#         band = int(round(band_ratio * self.T))

#         best = []
#         for L, plist in self.protos.items():
#             dmin = min(self.dtw_distance(r, p, band=band) for p in plist)
#             best.append((L, float(dmin)))
#         best.sort(key=lambda x: x[1])

#         best_lab, best_d = best[0]
#         second_d = best[1][1] if len(best) > 1 else 1e18
#         thr = float(self.thr_map.get(best_lab, best_d + 1.0))
#         margin_need = float(self.margin_map.get(best_lab, 0.0))
#         gap = second_d - best_d

#         if not self.use_reject:
#             return (best_lab, best) if want_stats else best_lab

#         ok = not ((best_d > thr) or (gap < margin_need))
#         if want_stats:
#             return (best_lab if ok else None, best, {"d0": best_d, "thr": thr, "gap": gap, "need": margin_need, "band": band_ratio})
#         return best_lab if ok else None

#     # --- GPT 폴백 (도움말 패널) ---
#     def _call_gpt_async(self, img_bgr, best_list):
#         if OPENAI_CLIENT is None:
#             self._gpt_busy = False
#             self.gpt_err = "OpenAI 클라이언트 없음(.streamlit/secrets.toml 확인)."
#             return

#         labels = list(self.protos.keys())
#         top5 = [{"label": L, "dtw": float(d)} for L, d in best_list[:5]]

#         sys_prompt = "너는 한국어 조수야. 사용자가 수어 인식을 시도했지만 모델이 확신하지 못했어."
#         user_prompt = (
#             "다음은 DTW 상위 후보와 거리값이야. "
#             "실시간 인식이 실패했을 때 사용자가 바로 개선할 수 있도록, "
#             "조명/프레이밍/속도/손가림/카메라 각도/배경 등에 대한 구체 체크리스트를 "
#             "3~6줄 한국어로 간단히 알려줘.\n"
#             f"가능한 라벨 집합: {labels}\n"
#             f"DTW top5: {top5}\n"
#             "불확실한 라벨 추정이 있다면 괄호에 (가능성 낮음)으로 표기해."
#         )

#         model_name = OPENAI_MODEL
#         use_vision = bool(self.use_gpt_vision and _supports_vision(model_name))

#         msgs = [{"role": "system", "content": sys_prompt}]
#         if use_vision and img_bgr is not None:
#             _, jpg = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
#             b64 = base64.b64encode(jpg.tobytes()).decode("utf-8")
#             msgs.append({
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": user_prompt},
#                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
#                 ]
#             })
#         else:
#             msgs.append({"role": "user", "content": user_prompt})
#             if self.use_gpt_vision and not use_vision:
#                 self.gpt_err = "현재 모델은 이미지 입력 미지원. 비전 옵션을 끄거나 gpt-5/4o/omni로 바꾸세요."

#         def _run():
#             try:
#                 resp = OPENAI_CLIENT.chat.completions.create(
#                     model=model_name,
#                     messages=msgs,
#                     temperature=0.3,
#                 )
#                 self.gpt_help = (resp.choices[0].message.content or "").strip()
#                 self.gpt_err = ""
#             except Exception as e:
#                 self.gpt_help = ""
#                 self.gpt_err = f"GPT 호출 오류: {e}"
#             finally:
#                 self._gpt_busy = False

#         threading.Thread(target=_run, daemon=True).start()

#     # --- GPT 보강문 (성공 시 한 문장) ---
#     def _augment_label_async(self, ko_label: str, stats: dict | None):
#         if OPENAI_CLIENT is None:
#             self._aug_busy = False
#             return
#         sys_p = "너는 한국어 음성 안내 문구를 간결하게 만들어 주는 조수야."
#         d0 = stats.get("d0") if stats else None
#         gap = stats.get("gap") if stats else None
#         user_p = (
#             f"인식된 수어: '{ko_label}'. "
#             "출력 규칙: 1문장, 6~14자, 존댓말, 과장/추측 금지, 마침표 포함. "
#             "예시: '곰입니다.', '공원이에요.' 처럼.\n"
#             f"(참고수치: d0={d0:.1f} gap={gap:.1f})" if (d0 is not None and gap is not None) else ""
#         )
#         def _run():
#             try:
#                 resp = OPENAI_CLIENT.chat.completions.create(
#                     model=OPENAI_MODEL,
#                     messages=[{"role":"system","content":sys_p},
#                               {"role":"user","content":user_p}],
#                     temperature=self.gpt_temp,
#                 )
#                 text = (resp.choices[0].message.content or "").strip()
#                 self.aug_text = text[:30]
#                 if self.gpt_speak and self.aug_text:
#                     # 보강문은 사용자 옵션(gpt_speak)에 따라 발화
#                     self.pending_tts.append(self.aug_text)  # 브라우저
#                     if _tts is not None:
#                         speak_async(self.aug_text)          # 서버
#             except Exception as e:
#                 self.aug_text = ""
#                 print("[GPT augment error]", e)
#             finally:
#                 self._aug_busy = False
#         threading.Thread(target=_run, daemon=True).start()

#     # --- 디버그 문자열 포맷(출력용) ---
#     def _format_debug_out(self, best, stats):
#         if not best: return ""
#         L0, d0 = best[0]
#         d1 = best[1][1] if len(best) > 1 else float("inf")
#         if stats:
#             return f"[{L0}] d0={d0:.3f} d1={d1:.3f} thr={stats['thr']:.3f} gap={stats['gap']:.3f} need={stats['need']:.3f} band={stats['band']:.2f}"
#         else:
#             band_ratio = self.band_override if self.band_override is not None else self.band_ratio
#             return f"[{L0}] d0={d0:.3f} d1={d1:.3f} band={band_ratio:.2f}"

#     # --- 발화 트리거(브라우저 + 서버) ---
#     def _maybe_speak(self, text: str):
#         if not self.speak or not text:
#             return
#         now = time.time()
#         if text == getattr(self, "_last_tts", "") and (now - self._last_tts_time) < self._tts_cooldown:
#             return
#         # 1) 브라우저 TTS 큐에 넣고 (메인 루프에서 실제 재생)
#         self.pending_tts.append(text)
#         # 2) 서버(pyTTSX3)도 가능하면 동시에
#         if _tts is not None:
#             speak_async(text)
#         self._last_tts = text
#         self._last_tts_time = now

#     # --- WebRTC 콜백 ---
#     def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#         img_bgr = frame.to_ndarray(format="bgr24")
#         h, w = img_bgr.shape[:2]

#         # 얼굴 탐지(밝기 보정)
#         self._cur_face = None
#         yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
#         yuv[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(yuv[:, :, 0])
#         bright = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
#         det_img = cv2.resize(bright, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)

#         face_results = self.face0.process(det_img[:, :, ::-1])
#         if not face_results.detections:
#             face_results = self.face1.process(det_img[:, :, ::-1])

#         face_bbox = None
#         if face_results.detections:
#             det = max(face_results.detections,
#                       key=lambda d: d.location_data.relative_bounding_box.width *
#                                     d.location_data.relative_bounding_box.height)
#             face_pts, face_bbox = self._face_keypoints_to_pixels(det, w, h)
#             self._cur_face = face_pts
#             if self.debug:
#                 for k in ("left_eye", "right_eye", "nose", "mouth"):
#                     p = face_pts[k].astype(int); cv2.circle(img_bgr, tuple(p), 3, (0,128,255), -1)
#                 x, y, bw, bh = face_bbox; cv2.rectangle(img_bgr, (x,y), (x+bw,y+bh), (0,128,255), 1)

#         # 손 탐지(얼굴 ROI 우선)
#         if face_bbox and self.use_roi:
#             x0, y0, rw, rh = self._expand_roi(face_bbox, w, h, *self.roi_scale)
#             crop = img_bgr[y0:y0+rh, x0:x0+rw]
#             res = self.hands.process(crop[:, :, ::-1])
#             left_xy, right_xy = None, None
#             if res.multi_hand_landmarks and res.multi_handedness:
#                 for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
#                     label_h = handed.classification[0].label
#                     xy = np.array([[x0 + p.x * rw, y0 + p.y * rh] for p in lm.landmark], dtype=np.float32)
#                     if label_h == "Left": left_xy = xy
#                     else: right_xy = xy
#                     if self.debug:
#                         self.mp.solutions.drawing_utils.draw_landmarks(
#                             img_bgr, lm, self.mp.solutions.hands.HAND_CONNECTIONS,
#                             landmark_drawing_spec=self.mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2),
#                             connection_drawing_spec=self.mp.solutions.drawing_utils.DrawingSpec(color=(0,200,0), thickness=1),
#                         )
#                         cv2.rectangle(img_bgr, (x0,y0), (x0+rw,y0+rh), (255,200,0), 1)
#             else:
#                 res = self.hands.process(img_bgr[:, :, ::-1])
#                 left_xy, right_xy = None, None
#                 if res.multi_hand_landmarks and res.multi_handedness:
#                     for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
#                         label_h = handed.classification[0].label
#                         xy = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)
#                         if label_h == "Left": left_xy = xy
#                         else: right_xy = xy
#                         if self.debug:
#                             self.mp.solutions.drawing_utils.draw_landmarks(img_bgr, lm, self.mp.solutions.hands.HAND_CONNECTIONS)
#         else:
#             res = self.hands.process(img_bgr[:, :, ::-1])
#             left_xy, right_xy = None, None
#             if res.multi_hand_landmarks and res.multi_handedness:
#                 for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
#                     label_h = handed.classification[0].label
#                     xy = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)
#                     if label_h == "Left": left_xy = xy
#                     else: right_xy = xy
#                     if self.debug:
#                         self.mp.solutions.drawing_utils.draw_landmarks(img_bgr, lm, self.mp.solutions.hands.HAND_CONNECTIONS)

#         # 세그먼트/분류(+거절, +연속확정, +GPT, +디버그 HUD)
#         ended = self._segment_step(left_xy, right_xy)
#         if ended:
#             out = self._classify(want_stats=True)
#             lab = out[0]
#             best = out[1]
#             stats = out[2] if len(out) > 2 else None
#             dbg_str = self._format_debug_out(best, stats)  # 🔹 항상 준비

#             # 디버그 텍스트(상단 HUD용)
#             if best:
#                 L0, d0 = best[0]
#                 d1 = best[1][1] if len(best) > 1 else float("inf")
#                 if stats:
#                     self.debug_info = (
#                         f"seg_end=1 act={self.act_count} idle={self.idle_count} active={self.active}\n"
#                         f"best={L0} d0={d0:.3f} d1={d1:.3f} thr={stats['thr']:.3f} "
#                         f"gap={stats['gap']:.3f} need={stats['need']:.3f} band={stats['band']:.2f}\n"
#                         f"decision={'ACCEPT' if lab else 'REJECT'} use_reject={self.use_reject}"
#                     )
#                 else:
#                     band_ratio = self.band_override if self.band_override is not None else self.band_ratio
#                     self.debug_info = (
#                         f"seg_end=1 act={self.act_count} idle={self.idle_count} active={self.active}\n"
#                         f"(loose) best={L0} d0={d0:.3f} d1={d1:.3f} band={band_ratio:.2f}"
#                     )

#             if lab is not None:
#                 # ACCEPT: 기존 로직 유지 (연속 2번일 때만 확정)
#                 self._nohit_streak = 0
#                 self.gpt_help = ""
#                 self._last_two.append(lab); self._last_two = self._last_two[-2:]
#                 if len(self._last_two) == 2 and self._last_two[0] == self._last_two[1]:
#                     ko = self.templates.get(lab, lab)
#                     if self.debug_out_mode == "label":
#                         out_text = ko
#                         self.last_display = ko
#                         self._maybe_speak(ko)
#                     elif self.debug_out_mode == "label+debug":
#                         out_text = f"{ko} ({dbg_str})" if dbg_str else ko
#                         self.last_display = out_text
#                         self._maybe_speak(ko)    # 음성은 라벨만
#                     else:  # "debug" (ACCEPT 시에도 디버그만)
#                         out_text = dbg_str or ko
#                         self.last_display = out_text
#                         self._maybe_speak(ko) 
#                         # self._maybe_speak(out_text[:80])  # 디버그 문장도 읽기

#                     self.last_label = ko
#                     self.text_out += (out_text + " ")
#                     self._last_two.clear()

#                     # GPT 보강문 (쿨다운)
#                     if self.use_gpt_enhance and not self._aug_busy and (time.time() - self._last_aug_time) >= 2.0:
#                         self._aug_busy = True
#                         self._last_aug_time = time.time()
#                         self._augment_label_async(self.last_label, stats)
#             else:
#                 # REJECT: 🔹 디버그만 모드에서는 디버그 문자열을 결과로 기록
#                 self._nohit_streak += 1
#                 if self.debug_out_mode == "debug" and dbg_str:
#                     self.last_display = dbg_str
#                     self.text_out += (dbg_str + " ")
#                     # self._maybe_speak(dbg_str[:80])
#                     self._maybe_speak(ko)
#                 if self.use_gpt and (self._nohit_streak >= 1) and not self._gpt_busy:
#                     now = time.time()
#                     if now - self._last_gpt_time >= 3.0:
#                         self._gpt_busy = True
#                         self._last_gpt_time = now
#                         self._call_gpt_async(img_bgr, best)

#         # 프레임 오버레이
#         if getattr(self, "last_display", ""):
#             msg = f"최근 인식: {self.last_display}"
#             img_bgr = draw_korean_text(img_bgr, msg, xy=(20, h-40), font_size=30, color=(0,255,0), stroke=2)
#         if self.debug and getattr(self, "debug_info", ""):
#             img_bgr = draw_korean_text(img_bgr, self.debug_info.split("\n")[0], xy=(20, 30), font_size=20, color=(255,255,0), stroke=2)

#         return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# # ---------------- Streamlit UI ----------------
# def main():
#     st.title("KSL 실시간 인식 — 모드 A (DTW 10단어)")

#     model = load_dtw_model()
#     templates = load_templates()
#     default_band = float(model.get('band_ratio', 0.15))

#     with st.sidebar:
#         st.header("옵션")
#         use_roi = st.checkbox("얼굴 주변 ROI 사용(권장)", value=True, key="opt_use_roi")
#         speak   = st.checkbox("인식 시 TTS 발화", value=True, key="opt_tts")
#         debug   = st.checkbox("디버그 오버레이(손/얼굴 표시)", value=True, key="opt_debug")
#         if st.button("모델 새로고침", key="refresh_model"):
#             st.cache_resource.clear(); st.rerun()
#         st.caption("브라우저 카메라 권한을 허용하세요. (브라우저 TTS는 탭 소리가 켜져 있어야 합니다)")

#         st.divider()
#         use_gpt = st.checkbox("인식 실패 시 GPT 도움말 사용", value=False, key="opt_gpt")
#         use_gpt_vision = st.checkbox("프레임 1장으로 GPT 비전 추정(실험)", value=False, key="opt_gpt_vision")
#         if use_gpt and OPENAI_CLIENT is None:
#             st.warning("`.streamlit/secrets.toml`에 OPENAI_API_KEY가 없습니다.")

#         st.divider()
#         apply_reject = st.checkbox("거절 규칙 적용(thr/margin)", value=True, key="opt_reject")
#         band_ratio_ui = st.slider("DTW band 비율(느슨↔엄격)", 0.05, 0.35, value=default_band, step=0.01, key="opt_band")

#         st.divider()
#         use_gpt_enh = st.checkbox("DTW 결과를 GPT로 보강", value=True, key="opt_gpt_enh")
#         gpt_say     = st.checkbox("보강문도 TTS", value=True, key="opt_gpt_enh_tts")
#         gpt_temp    = st.slider("보강문 창의성", 0.0, 1.0, value=0.2, step=0.05, key="opt_gpt_temp")

#         st.divider()
#         out_mode = st.selectbox(
#             "출력 형식",
#             ["일반(라벨만)", "라벨+디버그", "디버그만"],
#             index=0,
#             key="opt_out_mode"
#         )
#         mode_map = {"일반(라벨만)": "label", "라벨+디버그": "label+debug", "디버그만": "debug"}
#         debug_out_mode = mode_map[out_mode]

#     def factory():
#         return DTWProcessor(
#             model=model, templates=templates,
#             speak=speak, use_roi=use_roi, debug=debug,
#             use_gpt=use_gpt, use_gpt_vision=use_gpt_vision,
#             use_reject=apply_reject, band_override=band_ratio_ui,
#             use_gpt_enhance=use_gpt_enh, gpt_speak=gpt_say, gpt_temp=gpt_temp,
#             debug_out_mode=debug_out_mode,
#         )

#     ctx = webrtc_streamer(
#         key="ksl-mode-a-rtc",
#         mode=WebRtcMode.SENDRECV,
#         rtc_configuration=RTC_CONFIGURATION,
#         media_stream_constraints={"video": True, "audio": False},
#         async_processing=True,
#         video_processor_factory=factory,
#     )

#     col_left, col_mid, col_right = st.columns([2, 1, 1])
#     ph_text = col_left.empty()
#     ph_help = col_mid.empty()
#     ph_dbg  = col_right.empty()

#     while ctx and ctx.state.playing:
#         vp = ctx.video_processor
#         if vp:
#             # 🔹 실시간으로 옵션 반영 (재시작 없이)
#             vp.debug_out_mode = debug_out_mode
#             vp.use_reject     = apply_reject
#             vp.band_override  = band_ratio_ui
#             vp.use_roi        = use_roi
#             vp.debug          = debug
#             vp.speak          = speak
#             vp.use_gpt        = use_gpt
#             vp.use_gpt_vision = use_gpt_vision
#             vp.use_gpt_enhance= use_gpt_enh
#             vp.gpt_speak      = gpt_say
#             vp.gpt_temp       = gpt_temp

#             ph_text.markdown(f"### 인식 결과\n**{vp.text_out}**")

#             # 가운데 패널: 보강문 + 도움말 + 오류
#             aug = getattr(vp, "aug_text", "")
#             help_text = getattr(vp, "gpt_help", "")
#             err = getattr(vp, "gpt_err", "")
#             if aug or help_text or err:
#                 with ph_help.container():
#                     if aug:
#                         st.markdown("### 🤖 GPT 보강"); st.success(aug)
#                     if help_text:
#                         st.markdown("### 🛟 GPT 도움말"); st.info(help_text)
#                     if err:
#                         st.markdown("### ⚠️ GPT 오류"); st.error(err)
#             else:
#                 ph_help.markdown("### 🤖 GPT 보강 / 🛟 도움말\n_(없음)_")

#             # 디버그 텍스트
#             dbg = getattr(vp, "debug_info", "")
#             ph_dbg.markdown("### 🔍 디버그\n" + (f"```\n{dbg}\n```" if dbg else "_(없음)_"))

#             # 🔊 브라우저 TTS: 비디오 프로세서가 큐에 넣어둔 문장들을 실제로 말하게 함
#             if hasattr(vp, "pending_tts") and vp.pending_tts:
#                 while vp.pending_tts:
#                     to_say = vp.pending_tts.pop(0)
#                     browser_tts(to_say)

#         time.sleep(0.25)

#     if ctx and ctx.video_processor:
#         ph_text.markdown(f"### 인식 결과\n**{ctx.video_processor.text_out}**")
#         aug = getattr(ctx.video_processor, "aug_text", "")
#         help_text = getattr(ctx.video_processor, "gpt_help", "")
#         err = getattr(ctx.video_processor, "gpt_err", "")
#         if aug or help_text or err:
#             with ph_help.container():
#                 if aug:
#                     st.markdown("### 🤖 GPT 보강"); st.success(aug)
#                 if help_text:
#                     st.markdown("### 🛟 GPT 도움말"); st.info(help_text)
#                 if err:
#                     st.markdown("### ⚠️ GPT 오류"); st.error(err)
#         else:
#             ph_help.markdown("### 🤖 GPT 보강 / 🛟 도움말\n_(없음)_")

#         dbg = getattr(ctx.video_processor, "debug_info", "")
#         ph_dbg.markdown("### 🔍 디버그\n" + (f"```\n{dbg}\n```" if dbg else "_(없음)_"))

# if __name__ == "__main__":
#     main()


"""
Streamlit: DTW 10단어 실시간 인식 (모드 A)
- 얼굴+손 상대 피처 + DTW 거절(thr/margin) + ROI + 디버그 + 한글 렌더
- 인식 실패 연속 시 GPT 도움말(Secrets 기반) - 메인 결과와 분리
- 느슨 모드(거절 off) / band 비율 조절 / 디버그 HUD
- ✅ DTW 결과를 GPT로 ‘보강 문장’으로 함께 출력 (옵션)
- ✅ 보강 문장도 TTS 가능 (옵션)
- ✅ 출력 형식 선택: 라벨만 / 라벨+디버그 / 디버그만
- ✅ (fix) 디버그만 모드에서는 REJECT여도 디버그 문자열을 인식 결과에 기록
- ✅ (추가) 브라우저 TTS 지원: 서버(pyTTSX3) 불가 환경에서도 음성 출력
"""

import os, sys, time, json, base64, threading
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from streamlit.components.v1 import html as _st_html  # ★ 브라우저 TTS용
import av
from openai import OpenAI

# ---------------- Streamlit 페이지 설정 (첫 Streamlit 호출) ----------------
st.set_page_config(
    page_title="KSL DTW (모드 A)",
    layout="wide",
    page_icon="🤟",
    initial_sidebar_state="expanded",
)

# ---------------- 브라우저 TTS 유틸 ----------------
def browser_tts(text: str, rate: float = 1.0, pitch: float = 1.0, lang: str = "ko-KR"):
    """브라우저 음성합성(Web Speech API)로 말하기"""
    if not text:
        return
    safe = json.dumps(text)  # 따옴표/개행 이스케이프
    js = f"""
    <script>
    (function() {{
        try {{
            const u = new SpeechSynthesisUtterance({safe});
            u.lang = "{lang}";
            u.rate = {rate};
            u.pitch = {pitch};
            // 중복 발화 방지: 직전 발화 중단 후 재생
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(u);
        }} catch(e) {{ console.error(e); }}
    }})();
    </script>
    """
    _st_html(js, height=0)

# ---------------- OpenAI (secrets.toml) ----------------
def _S(name: str):
    import os, streamlit as st
    return os.environ.get(name) or st.secrets.get(name) or st.secrets.get("\ufeff"+name)

def _init_openai_from_secrets():
    try:
        from openai import OpenAI
        key  = _S("OPENAI_API_KEY")
        base = _S("OPENAI_BASE_URL")
        org  = _S("OPENAI_ORG")
        proj = _S("OPENAI_PROJECT")
        kw = {}
        if base: kw["base_url"] = base
        if org:  kw["organization"] = org
        if proj: kw["project"] = proj
        return OpenAI(api_key=key, **kw)
    except Exception as e:
        print("[OpenAI] secrets 초기화 실패 또는 키 없음:", e)
        return None

OPENAI_CLIENT = _init_openai_from_secrets()
OPENAI_MODEL  = _S("OPENAI_MODEL") or "gpt-4o"

def _supports_vision(model_name: str) -> bool:
    n = (model_name or "").lower()
    return any(k in n for k in ["gpt-5", "gpt-4o", "omni", "4.1"])

# ---------------- 경로 ----------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path: sys.path.append(str(HERE))
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

# ---------------- RTC 설정 ----------------
RTC_CONFIGURATION = {}

# ---------------- 한글 텍스트 렌더 (Pillow) ----------------
try:
    from PIL import Image, ImageDraw, ImageFont
    def draw_korean_text(bgr, text, xy=(20,30), font_size=28, color=(0,255,0), stroke=2, font_path=None):
        if font_path is None:
            for p in [
                r"C:\\Windows\\Fonts\\malgun.ttf",
                r"C:\\Windows\\Fonts\\malgunbd.ttf",
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            ]:
                if os.path.exists(p): font_path = p; break
        if not font_path or not os.path.exists(font_path): return bgr
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = ImageFont.truetype(font_path, font_size)
        draw.text(xy, text, font=font, fill=(color[2], color[1], color[0]),
                  stroke_width=stroke, stroke_fill=(0,0,0))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
except Exception:
    def draw_korean_text(bgr, text, xy=(20,30), font_size=28, color=(0,255,0), stroke=2, font_path=None):
        return bgr

# ---------------- TTS (서버 pyttsx3) ----------------
try:
    import pyttsx3
    _tts = pyttsx3.init()
    for v in _tts.getProperty("voices"):
        nid = (getattr(v, "id", "") or "").lower()
        nname = (getattr(v, "name", "") or "").lower()
        if "korean" in nname or "ko_" in nid or "ko-" in nid:
            _tts.setProperty("voice", v.id); break
    _tts.setProperty("rate", 170); _tts.setProperty("volume", 1.0)
except Exception:
    _tts = None

def speak_async(text: str):
    if not text or _tts is None: return
    def _run():
        try: _tts.say(text); _tts.runAndWait()
        except Exception: pass
    threading.Thread(target=_run, daemon=True).start()

# ---------------- 모델/템플릿 ----------------
@st.cache_resource
def load_dtw_model():
    import joblib
    return joblib.load(str(ROOT / "models" / "sign_dtw.pkl"))

@st.cache_resource
def load_templates():
    with open(HERE / "templates_ko.json", "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- 비디오 프로세서 ----------------
class DTWProcessor(VideoProcessorBase):
    """DTW + ROI + 얼굴/손 피처 + 리젝트 + GPT 도움말/보강 + 디버그 HUD + 브라우저/서버 TTS"""
    def __init__(self, model, templates, speak=True, use_roi=True, debug=True,
                 use_gpt=False, use_gpt_vision=False,
                 use_reject=True, band_override=None,
                 use_gpt_enhance=True, gpt_speak=True, gpt_temp=0.2,
                 debug_out_mode="label"):
        import mediapipe as mp
        self.mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.face0 = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)
        self.face1 = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
        self.drawing = mp.solutions.drawing_utils
        self.mp_fd = mp.solutions.face_detection

        self.model = model
        self.templates = templates
        self.protos = model["protos"]; self.T = int(model["T"])
        self.band_ratio = float(model.get("band_ratio", 0.15))
        self.thr_map = model.get("thr", {})
        self.margin_map = model.get("margin", {})

        self.speak = speak
        self.use_roi = use_roi
        self.debug = debug
        self.roi_scale = (1.8, 2.2)

        # 상태
        self.active = False; self.seq = []
        self.last_feat = None; self.idle_count = 0; self.act_count = 0
        self.text_out = ""; self.last_label = ""
        self.last_display = ""   # 프레임 오버레이 표시 문자열
        self._cur_face = None
        self._last_two = []  # 연속 확정용

        # GPT 폴백 상태
        self.use_gpt = use_gpt
        self.use_gpt_vision = use_gpt_vision
        self._nohit_streak = 0
        self._gpt_busy = False
        self._last_gpt_time = 0.0
        self.gpt_help = ""
        self.gpt_err = ""

        # GPT 보강(성공 시 한 문장)
        self.use_gpt_enhance = bool(use_gpt_enhance)
        self.gpt_speak = bool(gpt_speak)
        self.gpt_temp = float(gpt_temp)
        self.aug_text = ""
        self._aug_busy = False
        self._last_aug_time = 0.0

        # 리젝트/밴드
        self.use_reject = bool(use_reject)
        self.band_override = float(band_override) if band_override is not None else None

        # 출력 모드: "label" | "label+debug" | "debug"
        self.debug_out_mode = debug_out_mode

        # 프로토타입 차원
        any_lab = next(iter(self.protos)); any_proto = self.protos[any_lab][0]
        self.proto_dim = int(any_proto.shape[1])

        from sign_features import frame_feature, resample_seq, dtw_distance
        self.frame_feature = frame_feature
        self.resample_seq = resample_seq
        self.dtw_distance = dtw_distance

        self.debug_info = ""

        # 🔊 TTS 관련 상태
        self._last_tts = ""
        self._last_tts_time = 0.0
        self._tts_cooldown = 1.8  # 초: 같은 문구 연속 발화 방지
        self.pending_tts = []     # 브라우저 TTS 큐

    # --- 유틸 ---
    def _face_keypoints_to_pixels(self, det, w, h):
        kps = det.location_data.relative_keypoints
        def rp(idx):
            return np.array([kps[idx].x * w, kps[idx].y * h], dtype=np.float32)
        pts = {
            "right_eye": rp(self.mp_fd.FaceKeyPoint.RIGHT_EYE),
            "left_eye":  rp(self.mp_fd.FaceKeyPoint.LEFT_EYE),
            "nose":      rp(self.mp_fd.FaceKeyPoint.NOSE_TIP),
            "mouth":     rp(self.mp_fd.FaceKeyPoint.MOUTH_CENTER),
            "right_ear": rp(self.mp_fd.FaceKeyPoint.RIGHT_EAR_TRAGION),
            "left_ear":  rp(self.mp_fd.FaceKeyPoint.LEFT_EAR_TRAGION),
        }
        rel = det.location_data.relative_bounding_box
        bbox = (int(rel.xmin*w), int(rel.ymin*h), int(rel.width*w), int(rel.height*h))
        return pts, bbox

    def _expand_roi(self, bbox, w, h, sx=1.8, sy=2.2):
        x, y, bw, bh = bbox
        cx, cy = x + bw/2, y + bh/2
        nw, nh = int(bw*sx), int(bh*sy)
        nx, ny = int(cx - nw/2), int(cy - nh/2)
        nx = max(0, nx); ny = max(0, ny)
        nx2 = min(w, nx + nw); ny2 = min(h, ny + nh)
        return (nx, ny, nx2 - nx, ny2 - ny)

    # --- 세그먼트 ---
    def _segment_step(self, left_xy, right_xy):
        feat = self.frame_feature(left_xy, right_xy, face=self._cur_face)
        if feat.shape[0] != self.proto_dim:
            feat = self.frame_feature(left_xy, right_xy, face=None)

        if self.last_feat is None: self.last_feat = feat
        mv = float(np.linalg.norm(feat - self.last_feat)); self.last_feat = feat

        hand_present = (left_xy is not None) or (right_xy is not None)
        self.act_count = min(10, self.act_count + 1) if (hand_present and mv > 0.10) else max(0, self.act_count - 1)
        self.idle_count = min(10, self.idle_count + 1) if ((not hand_present) or mv < 0.03) else max(0, self.idle_count - 1)

        if not self.active and self.act_count >= 7:
            self.active = True; self.seq = []
        if self.active:
            self.seq.append(feat)
            if self.idle_count >= 7:
                self.active = False
                return True
        return False

    # --- 분류 (+상위 후보/지표) ---
    def _classify(self, want_stats=False):
        if not self.seq:
            return (None, []) if want_stats else None

        r = self.resample_seq(np.stack(self.seq), self.T)
        band_ratio = self.band_override if self.band_override is not None else self.band_ratio
        band = int(round(band_ratio * self.T))

        best = []
        for L, plist in self.protos.items():
            dmin = min(self.dtw_distance(r, p, band=band) for p in plist)
            best.append((L, float(dmin)))
        best.sort(key=lambda x: x[1])

        best_lab, best_d = best[0]
        second_d = best[1][1] if len(best) > 1 else 1e18
        thr = float(self.thr_map.get(best_lab, best_d + 1.0))
        margin_need = float(self.margin_map.get(best_lab, 0.0))
        gap = second_d - best_d

        if not self.use_reject:
            return (best_lab, best) if want_stats else best_lab

        ok = not ((best_d > thr) or (gap < margin_need))
        if want_stats:
            return (best_lab if ok else None, best, {"d0": best_d, "thr": thr, "gap": gap, "need": margin_need, "band": band_ratio})
        return best_lab if ok else None

    # --- GPT 폴백 (도움말 패널) ---
    def _call_gpt_async(self, img_bgr, best_list):
        if OPENAI_CLIENT is None:
            self._gpt_busy = False
            self.gpt_err = "OpenAI 클라이언트 없음(.streamlit/secrets.toml 확인)."
            return

        labels = list(self.protos.keys())
        top5 = [{"label": L, "dtw": float(d)} for L, d in best_list[:5]]

        sys_prompt = "너는 한국어 조수야. 사용자가 수어 인식을 시도했지만 모델이 확신하지 못했어."
        user_prompt = (
            "다음은 DTW 상위 후보와 거리값이야. "
            "실시간 인식이 실패했을 때 사용자가 바로 개선할 수 있도록, "
            "조명/프레이밍/속도/손가림/카메라 각도/배경 등에 대한 구체 체크리스트를 "
            "3~6줄 한국어로 간단히 알려줘.\n"
            f"가능한 라벨 집합: {labels}\n"
            f"DTW top5: {top5}\n"
            "불확실한 라벨 추정이 있다면 괄호에 (가능성 낮음)으로 표기해."
        )

        model_name = OPENAI_MODEL
        use_vision = bool(self.use_gpt_vision and _supports_vision(model_name))

        msgs = [{"role": "system", "content": sys_prompt}]
        if use_vision and img_bgr is not None:
            _, jpg = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            b64 = base64.b64encode(jpg.tobytes()).decode("utf-8")
            msgs.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            })
        else:
            msgs.append({"role": "user", "content": user_prompt})
            if self.use_gpt_vision and not use_vision:
                self.gpt_err = "현재 모델은 이미지 입력 미지원. 비전 옵션을 끄거나 gpt-5/4o/omni로 바꾸세요."

        def _run():
            try:
                resp = OPENAI_CLIENT.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    temperature=0.3,
                )
                self.gpt_help = (resp.choices[0].message.content or "").strip()
                self.gpt_err = ""
            except Exception as e:
                self.gpt_help = ""
                self.gpt_err = f"GPT 호출 오류: {e}"
            finally:
                self._gpt_busy = False

        threading.Thread(target=_run, daemon=True).start()

    # --- GPT 보강문 (성공 시 한 문장) ---
    def _augment_label_async(self, ko_label: str, stats: dict | None):
        if OPENAI_CLIENT is None:
            self._aug_busy = False
            return
        sys_p = "너는 한국어 음성 안내 문구를 간결하게 만들어 주는 조수야."
        d0 = stats.get("d0") if stats else None
        gap = stats.get("gap") if stats else None
        user_p = (
            f"인식된 수어: '{ko_label}'. "
            "출력 규칙: 1문장, 6~14자, 존댓말, 과장/추측 금지, 마침표 포함. "
            "예시: '곰입니다.', '공원이에요.' 처럼.\n"
            f"(참고수치: d0={d0:.1f} gap={gap:.1f})" if (d0 is not None and gap is not None) else ""
        )
        def _run():
            try:
                resp = OPENAI_CLIENT.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role":"system","content":sys_p},
                              {"role":"user","content":user_p}],
                    temperature=self.gpt_temp,
                )
                text = (resp.choices[0].message.content or "").strip()
                self.aug_text = text[:30]
                if self.gpt_speak and self.aug_text:
                    # 보강문은 사용자 옵션(gpt_speak)에 따라 발화
                    self.pending_tts.append(self.aug_text)  # 브라우저
                    if _tts is not None:
                        speak_async(self.aug_text)          # 서버
            except Exception as e:
                self.aug_text = ""
                print("[GPT augment error]", e)
            finally:
                self._aug_busy = False
        threading.Thread(target=_run, daemon=True).start()

    # --- 디버그 문자열 포맷(출력용) ---
    def _format_debug_out(self, best, stats):
        if not best: return ""
        L0, d0 = best[0]
        d1 = best[1][1] if len(best) > 1 else float("inf")
        if stats:
            return f"[{L0}] d0={d0:.3f} d1={d1:.3f} thr={stats['thr']:.3f} gap={stats['gap']:.3f} need={stats['need']:.3f} band={stats['band']:.2f}"
        else:
            band_ratio = self.band_override if self.band_override is not None else self.band_ratio
            return f"[{L0}] d0={d0:.3f} d1={d1:.3f} band={band_ratio:.2f}"

    # --- 디버그 문자열에서 라벨만 추출 ---
    def _extract_word_from_debug(self, dbg: str) -> str | None:
        """예: '[곰] d0=...' → '곰' 만 추출. 템플릿 매핑도 적용."""
        if not dbg:
            return None
        try:
            import re
            m = re.search(r"\[([^\[\]]+)\]", dbg)
            if not m:
                return None
            label = m.group(1).strip()
            return self.templates.get(label, label)
        except Exception:
            return None

    # --- 발화 트리거(브라우저 + 서버) ---
    def _maybe_speak(self, text: str):
        if not self.speak or not text:
            return
        now = time.time()
        if text == getattr(self, "_last_tts", "") and (now - self._last_tts_time) < self._tts_cooldown:
            return
        # 1) 브라우저 TTS 큐에 넣고 (메인 루프에서 실제 재생)
        self.pending_tts.append(text)
        # 2) 서버(pyTTSX3)도 가능하면 동시에
        if _tts is not None:
            speak_async(text)
        self._last_tts = text
        self._last_tts_time = now

    # --- WebRTC 콜백 ---
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        h, w = img_bgr.shape[:2]

        # 얼굴 탐지(밝기 보정)
        self._cur_face = None
        yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(yuv[:, :, 0])
        bright = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        det_img = cv2.resize(bright, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)

        face_results = self.face0.process(det_img[:, :, ::-1])
        if not face_results.detections:
            face_results = self.face1.process(det_img[:, :, ::-1])

        face_bbox = None
        if face_results.detections:
            det = max(face_results.detections,
                      key=lambda d: d.location_data.relative_bounding_box.width *
                                    d.location_data.relative_bounding_box.height)
            face_pts, face_bbox = self._face_keypoints_to_pixels(det, w, h)
            self._cur_face = face_pts
            if self.debug:
                for k in ("left_eye", "right_eye", "nose", "mouth"):
                    p = face_pts[k].astype(int); cv2.circle(img_bgr, tuple(p), 3, (0,128,255), -1)
                x, y, bw, bh = face_bbox; cv2.rectangle(img_bgr, (x,y), (x+bw,y+bh), (0,128,255), 1)

        # 손 탐지(얼굴 ROI 우선)
        if face_bbox and self.use_roi:
            x0, y0, rw, rh = self._expand_roi(face_bbox, w, h, *self.roi_scale)
            crop = img_bgr[y0:y0+rh, x0:x0+rw]
            res = self.hands.process(crop[:, :, ::-1])
            left_xy, right_xy = None, None
            if res.multi_hand_landmarks and res.multi_handedness:
                for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label_h = handed.classification[0].label
                    xy = np.array([[x0 + p.x * rw, y0 + p.y * rh] for p in lm.landmark], dtype=np.float32)
                    if label_h == "Left": left_xy = xy
                    else: right_xy = xy
                    if self.debug:
                        self.mp.solutions.drawing_utils.draw_landmarks(
                            img_bgr, lm, self.mp.solutions.hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=self.mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2),
                            connection_drawing_spec=self.mp.solutions.drawing_utils.DrawingSpec(color=(0,200,0), thickness=1),
                        )
                        cv2.rectangle(img_bgr, (x0,y0), (x0+rw,y0+rh), (255,200,0), 1)
            else:
                res = self.hands.process(img_bgr[:, :, ::-1])
                left_xy, right_xy = None, None
                if res.multi_hand_landmarks and res.multi_handedness:
                    for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                        label_h = handed.classification[0].label
                        xy = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)
                        if label_h == "Left": left_xy = xy
                        else: right_xy = xy
                        if self.debug:
                            self.mp.solutions.drawing_utils.draw_landmarks(img_bgr, lm, self.mp.solutions.hands.HAND_CONNECTIONS)
        else:
            res = self.hands.process(img_bgr[:, :, ::-1])
            left_xy, right_xy = None, None
            if res.multi_hand_landmarks and res.multi_handedness:
                for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label_h = handed.classification[0].label
                    xy = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)
                    if label_h == "Left": left_xy = xy
                    else: right_xy = xy
                    if self.debug:
                        self.mp.solutions.drawing_utils.draw_landmarks(img_bgr, lm, self.mp.solutions.hands.HAND_CONNECTIONS)

        # 세그먼트/분류(+거절, +연속확정, +GPT, +디버그 HUD)
        ended = self._segment_step(left_xy, right_xy)
        if ended:
            out = self._classify(want_stats=True)
            lab = out[0]
            best = out[1]
            stats = out[2] if len(out) > 2 else None
            dbg_str = self._format_debug_out(best, stats)  # 🔹 항상 준비

            # 디버그 텍스트(상단 HUD용)
            if best:
                L0, d0 = best[0]
                d1 = best[1][1] if len(best) > 1 else float("inf")
                if stats:
                    self.debug_info = (
                        f"seg_end=1 act={self.act_count} idle={self.idle_count} active={self.active}\n"
                        f"best={L0} d0={d0:.3f} d1={d1:.3f} thr={stats['thr']:.3f} "
                        f"gap={stats['gap']:.3f} need={stats['need']:.3f} band={stats['band']:.2f}\n"
                        f"decision={'ACCEPT' if lab else 'REJECT'} use_reject={self.use_reject}"
                    )
                else:
                    band_ratio = self.band_override if self.band_override is not None else self.band_ratio
                    self.debug_info = (
                        f"seg_end=1 act={self.act_count} idle={self.idle_count} active={self.active}\n"
                        f"(loose) best={L0} d0={d0:.3f} d1={d1:.3f} band={band_ratio:.2f}"
                    )

            if lab is not None:
                # ACCEPT: 기존 로직 유지 (연속 2번일 때만 확정)
                self._nohit_streak = 0
                self.gpt_help = ""
                self._last_two.append(lab); self._last_two = self._last_two[-2:]
                if len(self._last_two) == 2 and self._last_two[0] == self._last_two[1]:
                    ko = self.templates.get(lab, lab)
                    if self.debug_out_mode == "label":
                        out_text = ko
                        self.last_display = ko
                        self._maybe_speak(ko)
                    elif self.debug_out_mode == "label+debug":
                        out_text = f"{ko} ({dbg_str})" if dbg_str else ko
                        self.last_display = out_text
                        self._maybe_speak(ko)    # 음성은 라벨만
                    else:  # "debug" (ACCEPT 시에도 디버그만)
                        out_text = dbg_str or ko
                        self.last_display = out_text
                        self._maybe_speak(ko)    # 디버그만 모드여도 발화는 '라벨'만

                    self.last_label = ko
                    self.text_out += (out_text + " ")
                    self._last_two.clear()

                    # GPT 보강문 (쿨다운)
                    if self.use_gpt_enhance and not self._aug_busy and (time.time() - self._last_aug_time) >= 2.0:
                        self._aug_busy = True
                        self._last_aug_time = time.time()
                        self._augment_label_async(self.last_label, stats)
            else:
                # REJECT: 디버그만 모드에서는 디버그 문자열을 결과로 기록 + '단어만' 발화
                self._nohit_streak += 1

                if self.debug_out_mode == "debug" and dbg_str:
                    self.last_display = dbg_str
                    self.text_out += (dbg_str + " ")

                # 🔊 디버그 문자열에서 단어 추출 → TTS
                word = self._extract_word_from_debug(dbg_str)
                if word:
                    self._maybe_speak(word)
                elif best:
                    # 추출 실패 시 top-1 라벨로 폴백
                    fallback = self.templates.get(best[0][0], best[0][0])
                    self._maybe_speak(fallback)

                # GPT 도움말 트리거
                if self.use_gpt and (self._nohit_streak >= 1) and not self._gpt_busy:
                    now = time.time()
                    if now - self._last_gpt_time >= 3.0:
                        self._gpt_busy = True
                        self._last_gpt_time = now
                        self._call_gpt_async(img_bgr, best)

        # 프레임 오버레이
        if getattr(self, "last_display", ""):
            msg = f"최근 인식: {self.last_display}"
            img_bgr = draw_korean_text(img_bgr, msg, xy=(20, h-40), font_size=30, color=(0,255,0), stroke=2)
        if self.debug and getattr(self, "debug_info", ""):
            img_bgr = draw_korean_text(img_bgr, self.debug_info.split("\n")[0], xy=(20, 30), font_size=20, color=(255,255,0), stroke=2)

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# ---------------- Streamlit UI ----------------
def main():
    st.title("KSL 실시간 인식 — 모드 A (DTW 10단어)")

    model = load_dtw_model()
    templates = load_templates()
    default_band = float(model.get('band_ratio', 0.15))

    with st.sidebar:
        st.header("옵션")
        use_roi = st.checkbox("얼굴 주변 ROI 사용(권장)", value=True, key="opt_use_roi")
        speak   = st.checkbox("인식 시 TTS 발화", value=True, key="opt_tts")
        debug   = st.checkbox("디버그 오버레이(손/얼굴 표시)", value=True, key="opt_debug")
        if st.button("모델 새로고침", key="refresh_model"):
            st.cache_resource.clear(); st.rerun()
        st.caption("브라우저 카메라 권한을 허용하세요. (브라우저 TTS는 탭 소리가 켜져 있어야 합니다)")

        st.divider()
        use_gpt = st.checkbox("인식 실패 시 GPT 도움말 사용", value=False, key="opt_gpt")
        use_gpt_vision = st.checkbox("프레임 1장으로 GPT 비전 추정(실험)", value=False, key="opt_gpt_vision")
        if use_gpt and OPENAI_CLIENT is None:
            st.warning("`.streamlit/secrets.toml`에 OPENAI_API_KEY가 없습니다.")

        st.divider()
        apply_reject = st.checkbox("거절 규칙 적용(thr/margin)", value=True, key="opt_reject")
        band_ratio_ui = st.slider("DTW band 비율(느슨↔엄격)", 0.05, 0.35, value=default_band, step=0.01, key="opt_band")

        st.divider()
        use_gpt_enh = st.checkbox("DTW 결과를 GPT로 보강", value=True, key="opt_gpt_enh")
        gpt_say     = st.checkbox("보강문도 TTS", value=True, key="opt_gpt_enh_tts")
        gpt_temp    = st.slider("보강문 창의성", 0.0, 1.0, value=0.2, step=0.05, key="opt_gpt_temp")

        st.divider()
        out_mode = st.selectbox(
            "출력 형식",
            ["일반(라벨만)", "라벨+디버그", "디버그만"],
            index=0,
            key="opt_out_mode"
        )
        mode_map = {"일반(라벨만)": "label", "라벨+디버그": "label+debug", "디버그만": "debug"}
        debug_out_mode = mode_map[out_mode]

    def factory():
        return DTWProcessor(
            model=model, templates=templates,
            speak=speak, use_roi=use_roi, debug=debug,
            use_gpt=use_gpt, use_gpt_vision=use_gpt_vision,
            use_reject=apply_reject, band_override=band_ratio_ui,
            use_gpt_enhance=use_gpt_enh, gpt_speak=gpt_say, gpt_temp=gpt_temp,
            debug_out_mode=debug_out_mode,
        )

    ctx = webrtc_streamer(
        key="ksl-mode-a-rtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_processor_factory=factory,
    )

    col_left, col_mid, col_right = st.columns([2, 1, 1])
    ph_text = col_left.empty()
    ph_help = col_mid.empty()
    ph_dbg  = col_right.empty()

    while ctx and ctx.state.playing:
        vp = ctx.video_processor
        if vp:
            # 🔹 실시간으로 옵션 반영 (재시작 없이)
            vp.debug_out_mode = debug_out_mode
            vp.use_reject     = apply_reject
            vp.band_override  = band_ratio_ui
            vp.use_roi        = use_roi
            vp.debug          = debug
            vp.speak          = speak
            vp.use_gpt        = use_gpt
            vp.use_gpt_vision = use_gpt_vision
            vp.use_gpt_enhance= use_gpt_enh
            vp.gpt_speak      = gpt_say
            vp.gpt_temp       = gpt_temp

            ph_text.markdown(f"### 인식 결과\n**{vp.text_out}**")

            # 가운데 패널: 보강문 + 도움말 + 오류
            aug = getattr(vp, "aug_text", "")
            help_text = getattr(vp, "gpt_help", "")
            err = getattr(vp, "gpt_err", "")
            if aug or help_text or err:
                with ph_help.container():
                    if aug:
                        st.markdown("### 🤖 GPT 보강"); st.success(aug)
                    if help_text:
                        st.markdown("### 🛟 GPT 도움말"); st.info(help_text)
                    if err:
                        st.markdown("### ⚠️ GPT 오류"); st.error(err)
            else:
                ph_help.markdown("### 🤖 GPT 보강 / 🛟 도움말\n_(없음)_")

            # 디버그 텍스트
            dbg = getattr(vp, "debug_info", "")
            ph_dbg.markdown("### 🔍 디버그\n" + (f"```\n{dbg}\n```" if dbg else "_(없음)_"))

            # 🔊 브라우저 TTS: 비디오 프로세서가 큐에 넣어둔 문장들을 실제로 말하게 함
            if hasattr(vp, "pending_tts") and vp.pending_tts:
                while vp.pending_tts:
                    to_say = vp.pending_tts.pop(0)
                    browser_tts(to_say)

        time.sleep(0.25)

    if ctx and ctx.video_processor:
        ph_text.markdown(f"### 인식 결과\n**{ctx.video_processor.text_out}**")
        aug = getattr(ctx.video_processor, "aug_text", "")
        help_text = getattr(ctx.video_processor, "gpt_help", "")
        err = getattr(ctx.video_processor, "gpt_err", "")
        if aug or help_text or err:
            with ph_help.container():
                if aug:
                    st.markdown("### 🤖 GPT 보강"); st.success(aug)
                if help_text:
                    st.markdown("### 🛟 GPT 도움말"); st.info(help_text)
                if err:
                    st.markdown("### ⚠️ GPT 오류"); st.error(err)
        else:
            ph_help.markdown("### 🤖 GPT 보강 / 🛟 도움말\n_(없음)_")

        dbg = getattr(ctx.video_processor, "debug_info", "")
        ph_dbg.markdown("### 🔍 디버그\n" + (f"```\n{dbg}\n```" if dbg else "_(없음)_"))

if __name__ == "__main__":
    main()
