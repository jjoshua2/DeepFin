@echo off
REM DeepFin UCI entry point for Windows chess GUIs (Arena, CuteChess-cli,
REM BanksiaGUI, ChessBase). Point the GUI's "engine executable" field at
REM this .bat; stdin/stdout are forwarded through wsl.exe to the Python
REM UCI loop running inside WSL, where the CUDA + Python env live.
REM
REM === Configure your checkpoint ===
REM Either uncomment the `set` line below and put the WSL-side path to
REM trainer.pt, OR set DEEPFIN_CKPT as a Windows user environment variable
REM (System Properties -> Environment Variables). The WSLENV line forwards
REM whichever value is in scope into the WSL process as-is; GUI launches
REM are non-interactive shells, so we can't rely on .bashrc being sourced.
REM
REM set DEEPFIN_CKPT=/home/josh/ckpts/deepfin.pt
REM =================================
REM
REM If you use a non-default WSL distro, change `wsl.exe` below to
REM `wsl.exe -d <DistroName>` (example: `wsl.exe -d Ubuntu-22.04`).

setlocal EnableDelayedExpansion

REM Resolve the sibling bash wrapper's path and translate to WSL form.
for /f "usebackq delims=" %%P in (`wsl.exe wslpath "%~dp0deepfin"`) do set "WSL_PATH=%%P"
if "%WSL_PATH%"=="" (
    echo DeepFin: could not resolve WSL path for %~dp0deepfin 1>&2
    echo Is WSL installed and is scripts\deepfin present next to this file? 1>&2
    exit /b 1
)

REM Forward DEEPFIN_CKPT verbatim into WSL (no path translation — the
REM value is already a POSIX path inside WSL's filesystem).
set "WSLENV=DEEPFIN_CKPT"

wsl.exe -- "%WSL_PATH%" %*

endlocal
