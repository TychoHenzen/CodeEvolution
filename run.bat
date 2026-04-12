@echo off
setlocal enabledelayedexpansion
call .venv\Scripts\activate.bat
cd /d D:\rust-target\axiom2d

if exist .codeevolve\output (
    set /p RESUME="Checkpoint found. Resume from existing checkpoint? [Y/N]: "
    if /i "!RESUME!"=="Y" (
        codeevolve reinit
        codeevolve run
        goto :eof
    )
)

git checkout .
if exist .codeevolve\output rmdir /s /q .codeevolve\output
codeevolve reinit
codeevolve run
