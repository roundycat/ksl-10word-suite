# -*- coding: utf-8 -*-
"""
sign_features.py (통합)
- 손 랜드마크 정규화 피처(기존) + 얼굴-손 위치 피처(옵션)
- frame_feature(left_xy, right_xy, face=None)
    * face: {'nose','mouth','left_eye','right_eye','left_ear','right_ear'} (각 2D np.array)
    * face가 None이거나 키가 부족하면 얼굴 피처는 0으로 패딩되어 기존과 100% 호환
- resample_seq / dtw_distance: 기존 로직 유지
"""
import numpy as np
from config import SIGN_HAND_MODE, HAND_DOMINANCE, MIRROR_LEFT_TO_RIGHT

# ---- Mediapipe 인덱스 상수 ----
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17
THUMB_TIP = 4
INDEX_TIP = 8

# ========== 손 피처(기존 정규화) ==========
def _norm_one(lm_xy: np.ndarray) -> np.ndarray:
    """
    한 손(21,2) 좌표를 손목 기준/중지 MCP 길이 스케일 정규화 + 손바닥 축 정렬 + (옵션) 좌우 미러링.
    반환: (21,2)
    """
    wrist = lm_xy[WRIST]
    xy = lm_xy - wrist  # 손목 원점화

    # 스케일: 중지 MCP까지의 길이
    denom = np.linalg.norm(lm_xy[MIDDLE_MCP] - wrist) + 1e-6
    xy = xy / denom

    # 손바닥 축 정렬: INDEX_MCP ↔ PINKY_MCP를 x축에 가깝도록 회전
    v = lm_xy[PINKY_MCP] - lm_xy[INDEX_MCP]
    ang = np.arctan2(v[1], v[0])
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = (xy @ R.T)

    # (옵션) 지배손 기준 좌우 통일
    if MIRROR_LEFT_TO_RIGHT:
        thumb_x = xy[THUMB_TIP, 0]
        if HAND_DOMINANCE == "right" and thumb_x < 0:
            xy[:, 0] *= -1
        if HAND_DOMINANCE == "left" and thumb_x > 0:
            xy[:, 0] *= -1
    return xy.astype(np.float32)

# ========== 얼굴-손 위치 피처 ==========
def _face_norm_refs(face: dict):
    """
    face 딕셔너리에서 코를 원점, 눈간거리로 스케일 정규화하는 헬퍼.
    반환: (ok, nose(2,), eye_dist(float), refs: dict)
    refs: {'mouth','left_ear','right_ear'} 각각 정규화된 (2,)
    """
    try:
        nose = np.asarray(face["nose"], dtype=np.float32)
        le = np.asarray(face["left_eye"], dtype=np.float32)
        re = np.asarray(face["right_eye"], dtype=np.float32)
    except Exception:
        return False, None, None, {}

    eye_dist = float(np.linalg.norm(le - re)) + 1e-6
    def norm(p): return (np.asarray(p, np.float32) - nose) / eye_dist

    refs = {
        "mouth": norm(face.get("mouth", le)),
        "left_ear": norm(face.get("left_ear", le)),
        "right_ear": norm(face.get("right_ear", re)),
        "nose": np.array([0.0, 0.0], dtype=np.float32),  # 정규화 좌표계에서 코는 (0,0)
    }
    return True, nose, eye_dist, refs

def _hand_face_feats(orig_xy: np.ndarray, nose: np.ndarray, eye_dist: float, refs: dict) -> np.ndarray:
    """
    얼굴 정규화 좌표계에서 손-얼굴 관계(거리/좌우/손목 위치) 피처를 생성.
    orig_xy: 원본 픽셀 좌표(21,2)
    """
    if orig_xy is None:
        return np.zeros(8, dtype=np.float32)

    def norm(p): return (p - nose) / eye_dist

    WR, TH, IDX = WRIST, THUMB_TIP, INDEX_TIP
    n_w = norm(orig_xy[WR])
    n_i = norm(orig_xy[IDX])
    n_t = norm(orig_xy[TH])

    n_m = refs["mouth"]
    n_le = refs["left_ear"]
    n_re = refs["right_ear"]

    # 거리 + 좌/우 + 손목 위치 + 검지-손목 길이(정규화)
    return np.array([
        np.linalg.norm(n_i - n_m),            # 검지-입
        np.linalg.norm(n_t - n_m),            # 엄지-입
        np.linalg.norm(n_i - n_le),           # 검지-왼볼
        np.linalg.norm(n_i - n_re),           # 검지-오른볼
        np.sign(n_i[0]),                      # 코 기준 좌/우
        n_w[0], n_w[1],                        # 손목 위치(정규화)
        np.linalg.norm(n_i - n_w),            # 검지-손목 길이(정규화)
    ], dtype=np.float32)

# ========== 프레임 피처 ==========
def frame_feature(left_xy: np.ndarray, right_xy: np.ndarray, face: dict | None = None) -> np.ndarray:
    """
    입력:
      - left_xy/right_xy: (21,2) ndarray 또는 None (픽셀 좌표)
      - face: (옵션) {'nose','mouth','left_eye','right_eye','left_ear','right_ear'} 각 (2,)
    출력:
      - 1D feature vector (z-score 정규화)
        구성 = [정규화된 왼손(42), 정규화된 오른손(42), 양손 상호(2), 얼굴-왼손(8), 얼굴-오른손(8)]
    """
    # --- 모드에 따라 한 손만 사용 ---
    orig_left, orig_right = left_xy, right_xy
    if SIGN_HAND_MODE == "right":
        left_xy = None
    elif SIGN_HAND_MODE == "left":
        right_xy = None

    # --- 손 정규화 피처(기존) ---
    if left_xy is None:
        left_feat = np.zeros((21, 2), dtype=np.float32)
    else:
        left_feat = _norm_one(left_xy.astype(np.float32))

    if right_xy is None:
        right_feat = np.zeros((21, 2), dtype=np.float32)
    else:
        right_feat = _norm_one(right_xy.astype(np.float32))

    # --- 양손 상호 피처(있으면) ---
    inter = []
    if (left_xy is not None) and (right_xy is not None):
        inter.append(np.linalg.norm(left_feat[WRIST] - right_feat[WRIST]))
        inter.append(np.linalg.norm(left_feat[INDEX_TIP] - right_feat[INDEX_TIP]))
    else:
        inter.extend([0.0, 0.0])

    # --- 얼굴-손 위치 피처(옵션) ---
    face_left = np.zeros(8, dtype=np.float32)
    face_right = np.zeros(8, dtype=np.float32)
    if face is not None:
        ok, nose, eye_dist, refs = _face_norm_refs(face)
        if ok:
            # 주의: 얼굴-손은 원본 픽셀 좌표로 계산(손 정규화/회전과 무관)
            face_left = _hand_face_feats(orig_left, nose, eye_dist, refs)
            face_right = _hand_face_feats(orig_right, nose, eye_dist, refs)

    # --- 벡터 결합 & z-score ---
    feat = np.concatenate([
        left_feat.reshape(-1),
        right_feat.reshape(-1),
        np.asarray(inter, dtype=np.float32),
        face_left, face_right
    ]).astype(np.float32)

    m = feat.mean()
    s = feat.std() + 1e-6
    return (feat - m) / s

# ========== 시퀀스 보간 ==========
def resample_seq(arr: np.ndarray, T: int = 32) -> np.ndarray:
    """
    선형 보간으로 길이 T로 리샘플 (N,T_feat) -> (T,T_feat)
    """
    L = len(arr)
    if L == 0:
        return np.zeros((T, arr.shape[1]), dtype=np.float32)
    xs = np.linspace(0, L - 1, T)
    idx0 = np.floor(xs).astype(int)
    idx1 = np.minimum(idx0 + 1, L - 1)
    frac = (xs - idx0)[:, None]
    return (1 - frac) * arr[idx0] + frac * arr[idx1]

# ========== DTW ==========
def dtw_distance(A: np.ndarray, B: np.ndarray, band: int | None = None) -> float:
    """
    사코-치바 밴드 제약 DTW 거리 (속도 안정)
    """
    Ta, Tb = len(A), len(B)
    if band is None:
        band = max(Ta, Tb) // 4
    INF = 1e18
    D = np.full((Ta + 1, Tb + 1), INF, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, Ta + 1):
        j0 = max(1, i - band)
        j1 = min(Tb, i + band)
        for j in range(j0, j1 + 1):
            cost = np.linalg.norm(A[i - 1] - B[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[Ta, Tb])
