# Setup on Windows (Phase 0)

Do steps 1 to 4 yourself in PowerShell. Then start Claude Code and give it the Phase 0 prompt from `PROMPTS.md`.

## 1. Tools (one time)
```powershell
winget install -e --id Python.Python.3.12
winget install -e --id Git.Git
# Claude Code (native installer). If your org uses another method, use that.
irm https://claude.ai/install.ps1 | iex
```
Close and reopen PowerShell, then check: `py -3.12 --version`, `git --version`, `claude --version`.

Why Python 3.12: Box2D (the lander's physics) ships a prebuilt wheel for it. On 3.13 or newer pip may try to
build Box2D from source, which needs `swig` and a C++ compiler.

## 2. Project folder
Unzip this kit into e.g. `C:\dev\lunar-lander-reward-shaping`, then:
```powershell
cd C:\dev\lunar-lander-reward-shaping
git init
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## 3. Dependencies
Claude Code will write these into `requirements.txt` in Phase 0. Install them now:
```powershell
pip install "gymnasium[box2d]>=1.2,<1.4" "stable-baselines3>=2.9,<3" "torch>=2.8" "moviepy>=1.0.3" "tensorboard>=2.16" "pytest>=8" "ruff>=0.6"
```
Why these ranges (checked October 2026):
- Stable-Baselines3 2.9.0 (June 2026) needs Python 3.10+, PyTorch 2.8+ and gymnasium below 2.0. It names gymnasium 1.3 as supported.
- gymnasium 1.4.0 came out on 2026-10-05, after SB3's last release. The `<1.4` cap avoids being the first to hit an incompatibility.
- PyTorch from PyPI on Windows is the CPU build. That is what we want: no graphics card needed (about 200 MB download).

If the Box2D install fails with an error mentioning `swig`:
```powershell
pip install swig
pip install "gymnasium[box2d]>=1.2,<1.4"
```
If it then asks for a C++ compiler: install "Microsoft C++ Build Tools", or move the project to WSL2.

## 4. Smoke test
```powershell
python -c "import Box2D, gymnasium as gym, stable_baselines3 as sb3, torch; e=gym.make('LunarLander-v3'); o,_=e.reset(seed=0); print(o.shape, gym.__version__, sb3.__version__, torch.__version__); print('SMOKE TEST OK')"
```
Expected: `(8,)`, three version numbers, then `SMOKE TEST OK`. Copy the versions into `docs/EXPERIMENTS.md`.

## 5. Start Claude Code
```powershell
claude
```
Claude Code reads `CLAUDE.md` automatically at startup. Paste the Phase 0 prompt from `docs/PROMPTS.md`.

Note: on Windows, Claude Code runs shell commands in Git Bash. There the venv's Python is `.venv/Scripts/python`.
`CLAUDE.md` already tells it to use that path.
