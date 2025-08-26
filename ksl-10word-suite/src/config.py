# -*- coding: utf-8 -*-


# 10개 한국어 수어(고정 어휘) 프리셋


# SIGN_LABELS = ["공장", "놀이터", "공원", "공사장", "곰"]
SIGN_LABELS = None  # 또는 사용 라벨들을 모두 포함

# 카메라 설정


CAM_INDEX = 0


FRAME_SIZE = (960, 540)





# 구문 인식 손 모드: right/left/both


SIGN_HAND_MODE = "both"





# 지화(KNN)용 설정(옵션: 본 패키지는 DTW 중심)


HAND_DOMINANCE = "right"      # 왼손 사용자면 "left"


MIRROR_LEFT_TO_RIGHT = True

# ===== Realtime ㄱ 인식 설정 =====
DEBUG_DRAW = True
USE_ROI_FOR_HANDS = True

WINDOW = 15           # 최근 N프레임 슬라이딩 윈도우
TRIGGER_COUNT = 9     # N 중 최소 몇 프레임 이상 조건 만족 시 인식
COOLDOWN_SEC = 1.2    # 같은 단어 재발화 쿨다운(초)

# 얼굴 기준 정규화 후 휴리스틱 임계값
GIYUK_THRESHOLDS = {
    "ANGLE_MIN": 60.0,     # 엄지-검지 각도
    "ANGLE_MAX": 120.0,
    "IDX_EXTENDED_MIN": 0.35,  # 손목→검지팁 길이
    "MID_EXTENDED_MAX": 0.33,  # 손목→중지팁 길이
    "NEAR_FACE_MAX": 1.2,      # 검지팁↔입중심 거리
}