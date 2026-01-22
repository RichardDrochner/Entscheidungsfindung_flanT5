@echo off
setlocal

REM Projektverzeichnis setzen
cd /d "%~dp0"

if not exist .venv (
  py -3.11 -m venv .venv
)

call .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

if not exist "models\flan-t5-base\config.json" (
  echo [INFO] Lade flan-t5-base...
  hf download google/flan-t5-base ^
    --local-dir models\flan-t5-base ^
    --max-workers 8
)
  if errorlevel 1 (
    echo [FEHLER] Download fehlgeschlagen.
    pause
    exit /b 1
  )
)

python app.py

pause
