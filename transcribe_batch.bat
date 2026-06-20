@echo off

if not exist "%~dp0.venv\Scripts\activate.bat" (
    echo Criando ambiente virtual...
    python -m venv "%~dp0.venv"
)

call "%~dp0.venv\Scripts\activate.bat"

python -c "import faster_whisper" 2>nul
if errorlevel 1 (
    echo Instalando faster-whisper...
    pip install faster-whisper
)

python "%~dp0transcribe_batch.py"
pause
