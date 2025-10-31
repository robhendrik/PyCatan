# PyCatan

PyCatan is a Python project containing tools, tutorials, and RL (reinforcement learning) players for experimenting with AI players for the board game "Catan".

This repository includes:

- `src/Py_Catan_AI/` — main package with game simulation, players, RL models, utilities, plotting and tutorial helpers.
- `docs/` — Jupyter notebooks with step-by-step tutorials (data generation, training, evaluation, tournaments).
- `scripts/` — small runnable helpers referenced by the notebooks and tutorials.
- `bootstrap/` — bundled model and data artifacts used for bootstrapping training and evaluation.
- `venvs/` — (local) virtual environments used for development (ignored by `.gitignore`).

Author: Rob Hendriks
Version: 1.0.0

## Quick start (Windows PowerShell)

1. Create and activate a virtual environment (recommended path: `venvs/env_pycatan`):

```powershell
python -m venv venvs\env_pycatan
venvs\env_pycatan\Scripts\Activate.ps1
```

2. Install development dependencies (example):

```powershell
pip install -r requirements.txt
# If requirements.txt is not present, typical packages include:
# pip install numpy pandas matplotlib seaborn tensorflow keras scikit-learn boto3 jupyter
```

3. Run a tutorial notebook

- Open `docs/` in JupyterLab/Notebook and run the notebooks. The top cell in the tutorials adds the `src/` directory to `sys.path` automatically (it searches upward from the current working directory). Start the notebook server from the repository root for best results:

```powershell
jupyter lab
```

4. Run tutorial as a script

Some tutorials have script equivalents under `scripts/`. Example:

```powershell
python .\scripts\PyCatan_tutorial_game_script.py
```

