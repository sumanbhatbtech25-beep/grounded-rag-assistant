@echo off
REM One-click setup + launch on Windows
cd /d "%~dp0"
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo Installing / checking dependencies...
pip install -q -r requirements.txt
python cli.py ingest
streamlit run app.py
