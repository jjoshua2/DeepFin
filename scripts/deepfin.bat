@echo off
REM DeepFin UCI entry point for Windows chess GUIs (Arena, CuteChess-cli,
REM BanksiaGUI, ChessBase). Point the GUI's "engine executable" field at
REM this .bat; stdin/stdout are forwarded through wsl.exe to the Python
REM UCI loop running inside WSL, where the CUDA + Python env live.
REM
REM Resolves its own location via %~dp0 and translates to a WSL path with
REM wslpath, so the same .bat works from any checkout location — no
REM hardcoded paths to edit.
REM
REM Prereqs:
REM   1. WSL2 installed, with the chess-anti-engine repo cloned in WSL
REM      (WSL filesystem, not /mnt/c — faster and avoids permissions issues).
REM   2. `scripts/deepfin` (bash wrapper) is executable: chmod +x scripts/deepfin
REM   3. DEEPFIN_CKPT env var set inside WSL (e.g. in ~/.bashrc) pointing at
REM      your trainer.pt. The bash wrapper reads it.
REM
REM If you use a non-default WSL distro, change `wsl.exe` to
REM `wsl.exe -d <DistroName>`. Example: `wsl.exe -d Ubuntu-22.04`.

setlocal EnableDelayedExpansion

REM Resolve this .bat's directory and the sibling bash wrapper, then
REM convert to a WSL path. %~dp0 ends with a backslash; wslpath handles it.
for /f "usebackq delims=" %%P in (`wsl.exe wslpath "%~dp0deepfin"`) do set "WSL_PATH=%%P"

if "%WSL_PATH%"=="" (
    echo DeepFin: could not resolve WSL path for %~dp0deepfin 1>&2
    echo Is WSL installed and is scripts\deepfin present next to this file? 1>&2
    exit /b 1
)

wsl.exe -- "%WSL_PATH%" %*

endlocal
