# showingpromise

## Install
Run one installer from the project folder:

- PowerShell (recommended): `.\install.ps1`
- Command Prompt: `install.bat`

The installer:

1. Creates `.venv` (if needed)
2. Ensures `pip` exists in that environment
3. Lets you choose one dependency profile: `GPU (DirectML)`, `GPU (CUDA)`, or `CPU`
4. Installs the matching requirements file

## Run
Use the project virtual environment:

`.\.venv\Scripts\python.exe main.py`

## Optional non-interactive install
Pass a profile directly:

- `.\install.ps1 -Profile directml`
- `.\install.ps1 -Profile cuda`
- `.\install.ps1 -Profile cpu`
