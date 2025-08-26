# KSL 10-Word Suite — 발표용 요약(모드 A)

> **한 줄 요약**: 웹캠 + MediaPipe로 얼굴·손 랜드마크를 뽑아 **프레임 피처(손 정규화 + 얼굴-손 관계)**를 만들고, DTW 프로토타입과의 거리로 10개 수어를 실시간 분류합니다. **거절 규칙**, **ROI 가속**, **TTS**, **GPT 보강/도움말**을 포함한 Streamlit 데모까지 한 번에 구성.

>**AI Hub  수어 데이터 사용:**
https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=&topMenu=&srchOptnCnd=OPTNCND001&searchKeyword=%EC%88%98%EC%96%B4&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=103
---

## 1) 전체 구조
```
[데이터 수집]
 ├─ sign_capture.py (웹캠)
 └─ ingest_signs_csv.py (동영상+CSV → JSONL)
        ↓ frames:{left[21×2], right[21×2], face{kps}}
[피처]
 └─ sign_features.py → frame_feature()  # 손 정규화 + 얼굴-손 위치
        ↓ 시퀀스 resample(T=32)
[학습]
 └─ sign_train_dtw.py → 프로토타입(k개) + thr/margin + band_ratio
        ↓ models/sign_dtw.pkl
[실시간]
 ├─ ui_streamlit_sign.py (WebRTC, ROI, 거절, 디버그, TTS, GPT)
 └─ sign_infer_realtime.py (ㄱ 손모양 휴리스틱 + TTS)
```

---

## 2) 파일별 역할(요점)
- **config.py**: 라벨 프리셋, 카메라/해상도, 손 모드, ㄱ 인식 임계값.
- **sign_capture.py**: 웹캠에서 한 세그먼트 녹화 → `label.jsonl`에 append 저장.
- **ingest_signs_csv.py**: CSV(영상경로, 라벨, 구간) 일괄 변환 → 얼굴 ROI 기반 손 탐지 포함.
- **sign_features.py**: 프레임 피처(손 정규화 42×2, 양손 상호 2, 얼굴-손 8×2) + z-score.
- **sign_train_dtw.py**: 증강 → 피처화 → 프로토타입 선택(k) → 라벨별 **임계/여유** 계산 → 저장.
- **ui_streamlit_sign.py**: Streamlit WebRTC. 얼굴 ROI 가속, 거절(thr/margin), 밴드 조절, 디버그 HUD, 브라우저/서버 TTS, GPT 도움말/보강.
- **sign_infer_realtime.py**: ㄱ(기역) 손모양 전용 휴리스틱 + 슬라이딩 윈도우 투표 + TTS.
- **templates_ko.json**: 라벨→짧은 한국어 문장 템플릿(보강/발화용).

---

## 3) 프레임 피처 설계(핵심)
**손 정규화(각 손 21×2 → 42)**
- 손목 원점화 → **중지 MCP 길이**로 스케일 정규화 → **손바닥 축 정렬**(INDEX_MCP↔PINKY_MCP) → 지배손 기준 **좌우 미러링(옵션)**.

**얼굴-손 위치(손당 8, 총 16)**
- 코를 원점, **눈간거리**로 스케일 정규화 좌표계.
- 거리: 검지↔입, 엄지↔입, 검지↔왼/오른볼, **좌/우 부호(sign)**, 손목 (x,y), 검지↔손목 길이.

**전체 차원**: 왼손 42 + 오른손 42 + 양손상호 2 + 얼굴-왼손 8 + 얼굴-오른손 8 = **102차원**/프레임 → **z-score**.

**시퀀스**: 선형 보간으로 길이 **T=32**로 맞춤, DTW 입력.

---

## 4) DTW 분류 & 거절 로직
- **프로토타입 선택**: 라벨별로 전체 시퀀스 평균에 가까운 샘플을 **k개(기본 3)** 선택.
- **DTW**: 사코-치바 **band_ratio(기본 0.15)** 제약.
- **라벨별 임계(thr)**: intra-label 거리의 **IQR 기반 상한**(Q75 + 1.5×IQR).
- **여유(margin)**: 1·2위 거리 차의 **25퍼센타일**. `best_d>thr`이거나 `gap<margin`이면 **REJECT**.
- **평가**: acc(정확도)·coverage(수용율) 출력.

---

## 5) 실시간 데모(UI)
**카메라 처리**
- CLAHE 밝기 보정 → 얼굴 검출(두 모델 순차 시도) → **얼굴 박스 확장 ROI**에서 손 우선 검출.

**세그먼트 검출**
- 프레임 간 피처 이동량·손 존재로 **활성/휴지 상태** 전이 → 휴지 감지 시 세그먼트 종료.

**결정 안정화**
- **연속 2회 동일 라벨**일 때만 확정. (지터 감소)

**옵션(사이드바)**
- ROI 사용, TTS 발화, 디버그 오버레이, 거절 on/off, **band 비율 슬라이더**, 출력 형식(라벨만/라벨+디버그/디버그만).

**GPT 연동(선택)**
- **도움말 폴백**: REJECT가 이어지면 **DTW 상위 5 후보+거리**를 GPT에 전달 → 촬영 개선 체크리스트.
- **보강 문장**: 확정 라벨을 **짧은 한국어 문장**으로 다듬어 TTS(브라우저/서버) 가능.

---

## 6) ㄱ(기역) 휴리스틱 스크립트
- 얼굴 정규화 좌표계에서 **엄지-검지 각도**(60~120°), **검지 연장 길이**(min), **중지 길이**(max), **입 근접성**(max) 4조건.
- **슬라이딩 윈도우**(N=15) 다수결로 트리거 → "기역" 발화.
- 디버그 HUD: 각도/길이/거리 수치, ROI, 랜드마크.

---

## 7) 실행 순서(데모 기준)
```bash
# (권장) Python 3.11 가상환경
pip install -r requirements.txt

# 1) CSV→JSONL 변환
python ingest_signs_csv.py --csv meta10.csv --out_dir data/signs/jsonl

# 2) 학습(프로토타입/임계 저장)
python sign_train_dtw.py --data data/signs/jsonl --out models/sign_dtw.pkl --T 32 --k 3

# 3) 실시간 데모(브라우저 권한 허용)
streamlit run ui_streamlit_sign.py

# (옵션) ㄱ 휴리스틱 단독 데모
python sign_infer_realtime.py
```

---

## 8) 발표 데모 흐름(2~3분 스크립트)
1) **문제정의**: "적은 데이터로도 동작하는 수어 인식" 목표. 손 모양뿐 아니라 **얼굴 기준 위치**를 함께 쓰자.
2) **피처**: 손 정규화 + 얼굴-손 8개 관계값(입/볼/좌우/손목). 프레임당 102차원.
3) **DTW+거절**: 프로토타입과 거리, 라벨별 임계·여유로 **오인식보다 거절**을 우선.
4) **데모**: ROI·디버그·밴드 슬라이더를 보여주고, 두 단어를 제스처. (연속 2회 확정, TTS)
5) **실패 사례**: 일부러 실패 → **GPT 도움말**(촬영 체크리스트) 표시.
6) **확장**: ㄱ 휴리스틱 / 라벨 추가 / 딥러닝 교체 가능 포인트.

---

## 9) 트러블슈팅
- **카메라 안 잡힘**: OS 권한/장치 인덱스 확인(`config.CAM_INDEX`), DirectShow 우선.
- **한글 깨짐**: 시스템 글꼴(맑은 고딕 등) 경로 자동 탐색. 필요시 `draw_korean_text()`에 폰트 지정.
- **미디어파이프 설치 실패**: Python 3.11 권장(너무 최신 버전은 휠 부족).
- **OpenAI 키 없음**: GPT 옵션 off 또는 `.streamlit/secrets.toml`에 키 설정.

---

## 10) 핵심 파라미터(발표 슬라이드용)
- `T=32`, `k=3`, `band_ratio≈0.15`, `thr=Q75+1.5×IQR`, `margin=P25(2위-1위)`
- 세그먼트: `act_count≥7` 시작 / `idle_count≥7` 종료, **2회 연속 확정**
- ㄱ 휴리스틱: 각도·길이·근접 임계 + **N=15, 9표 이상** 트리거

---

## 11) 부록: 데이터 스키마(JSONL 예)
```json
{
  "label": "곰",
  "user": "u01",
  "fps": 30,
  "frames": [
    {"left": [[x,y],...], "right": [[x,y],...], "face": {"nose":[x,y], ...}},
    ...
  ],
  "video": "path/to.mp4",
  "range": [start,end]
}
```

