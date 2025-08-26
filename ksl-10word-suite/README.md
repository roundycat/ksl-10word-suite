# KSL 10-Word Suite (최종본)

**10개 한국어 수어 영상**만으로 **학습→실시간 데모**까지 바로 가능한 최소 패키지입니다.
라벨은 `src/config.py`의 `SIGN_LABELS`로 고정(기본 10개).

## 1) 설치
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) CSV 작성(10개 영상)
`meta10.csv` 템플릿을 복사해 **영상 경로/라벨/구간(start,end)**을 채워주세요.
- 라벨은 `src/config.py`의 10개 중에서만 사용하세요.
- 구간을 모르겠으면 `start,end`를 공란으로 두면 전체 구간을 사용합니다.

## 3) 변환 → 학습 → 데모
```bash
# 1) CSV → JSONL
python manage.py ingest-csv --csv meta10.csv --out_dir data/signs/jsonl
# 2) 학습
python manage.py train --data data/signs/jsonl --out models/sign_dtw.pkl --T 32 --k 3
# 3) 실시간 데모(또는 Streamlit)
python manage.py demo
# 또는
python manage.py ui
```

## 4) 폴더 구조
```
ksl-10word-suite/
 ├─ data/signs/jsonl/        # 변환 결과(JSONL; 한 줄=한 세그먼트)
 ├─ models/sign_dtw.pkl      # 학습 결과
 ├─ src/
 │   ├─ config.py
 │   ├─ templates_ko.json
 │   ├─ sign_features.py
 │   ├─ sign_capture.py
 │   ├─ sign_train_dtw.py
 │   ├─ sign_infer_realtime.py
 │   └─ ingest_signs_csv.py
 ├─ manage.py
 ├─ requirements.txt
 ├─ Makefile
 └─ meta10.csv               # 템플릿(편집해서 사용)
```

## 5) 환경/설정 팁
- 카메라: 40–60cm, 단색 배경, 밝게.
- 왼손 사용자면 `src/config.py`에서 `HAND_DOMINANCE="left"`로 변경(재학습 불요).
- 세그먼트가 너무 길거나 짧으면 CSV에서 start/end를 조정.
- 라벨을 바꾸고 싶으면 `src/config.py`와 `src/templates_ko.json`을 함께 수정하세요.

행운을 빌어요! 🚀
