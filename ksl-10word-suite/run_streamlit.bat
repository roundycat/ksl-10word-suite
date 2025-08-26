@echo off
SETLOCAL

REM --- 가상환경 자동 활성화(있으면) ---
IF EXIST ".venv311\Scripts\activate.bat" (
  CALL ".venv311\Scripts\activate.bat"
) ELSE IF EXIST ".venv\Scripts\activate.bat" (
  CALL ".venv\Scripts\activate.bat"
)

REM --- (선택) 콘솔 UTF-8 ---
chcp 65001 >NUL

REM --- 의존성 설치 ---
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM --- 모드 A(Streamlit) 실행 ---
python -m streamlit run src\ui_streamlit_sign.py

ENDLOCAL
