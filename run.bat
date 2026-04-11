call .venv\Scripts\activate.bat
cd /d D:\rust-target\axiom2d
git checkout .
if exist .codeevolve\output rmdir /s /q .codeevolve\output
codeevolve reinit
codeevolve run
