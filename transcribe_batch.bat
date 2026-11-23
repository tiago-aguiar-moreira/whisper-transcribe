@echo off

if not exist "%~dp0.venv\Scripts\activate.bat" (
    echo Criando ambiente virtual...
    python -m venv "%~dp0.venv"
)

call "%~dp0.venv\Scripts\activate.bat"

rem Instala/corrige as dependencias nas versoes fixadas (rapido quando ja estao ok)
pip install -q -r "%~dp0requirements.txt"

python "%~dp0transcribe_batch.py" %*
pause
