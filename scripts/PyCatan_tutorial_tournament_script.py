# %% [markdown]
# # PyCatan Tutorial 2

# %%
from pathlib import Path
import sys
import os
from IPython import get_ipython

# Ensure local `src/` is on sys.path so we can import the package from the notebook
nb_dir = Path().resolve()
# Add repo-relative src folders (not permanent)
sys.path.insert(0, str(nb_dir / ".." / "src"))
sys.path.insert(0, str(nb_dir / "src"))

try:
    import Py_Catan_AI as pc
except Exception as exc:
    raise ImportError(
        "Could not import `Py_Catan_AI`. Make sure the project's `src/` directory is present and dependencies are installed.\n"
        "If you are running the notebook from a different working directory, try setting the notebook's working directory to the repository root."
    ) from exc

# Standard data-science imports (convenience)
import numpy as np
import pandas as pd
import pickle
import boto3
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Convenience: access IPython when available
ip = get_ipython()

# %%
# Set working directory to repo root
os.chdir(pc.get_repo_root(target_name="PyCatan"))

# Determine if running in Jupyter Notebook or as a script and load args accordingly
if get_ipython():
    print("Running in Jupyter Notebook")
    args = pc.Args()
else:
    print("Running as Script")
    args = pc.parse_args()

# %% [markdown]
# # PyCatan game tutorial: Tournaments
# 
# We play a tournament with four players. To remove the effect of player order we rotate the order during the tournament. If we play 24 games every possible permutation of player order will have been used once, so we typically use multiples of 24 for the number of games.
# 
# The game outcome is determined by the number of victory points earned. The game stops when one player reaches 8 victory points. (Victory points: town = 2 VP, village = 1 VP, longest street = 2 VP.) To measure player performance we assign points per game: in total 17 points are awarded (10 for the winner, 5 for second place, 2 for third, and 0 for last). If all players are equal we expect, on average, that they each score 4.25 points. At the end of the tournament we also report the standard deviation of the points. This helps determine whether a player is statistically better for the given number of games, or whether observed differences could be caused by chance. A strong player typically outperforms others by 2–3 standard deviations.
# 
# ### The tutorials
# 
# - Running a game, including generating a game visualization/video with players speaking in character
# - Running a tournament
# - Data generation for bootstrapping a model
# - Model training for bootstrapping reinforcement learning
# - Reinforcement learning to train a model
# - Running an evaluation tournament to check the performance of a model
# 
# ### Directory structure
# 
# <pre>
# PyCatan/
# ├── docs/ # contains tutorials in Jupyter Notebook format
# ├── scripts/ # contains runnable helper scripts
# ├── bootstrap/ # contains data and models used for bootstrapping RL learning
# └── src/
#     └── Py_Catan_AI/
#         ├── models/ # contains the 'default' models used when creating a new instance
#         ├── data/ # contains some default data
#         └── visuals/ # contains some images used for visualization
# </pre>
# 
# ### Results for a tournament with 4 value-based players which should perform equally well (24 games)
# 
# We see the players score equally within the expected variation.
# 
# Overall tournament results:
# 
# | Player | Avg Points | Std Points | Avg Victory Pts | Std Victory Pts | Avg Rounds | Std Rounds |
# |---|---:|---:|---:|---:|---:|---:|
# | ValueBased1 | 4.19 | 0.80 | 5.79 | 0.37 | 16.38 | 0.94 |
# | ValueBased2 | 4.09 | 0.71 | 5.79 | 0.33 | 16.38 | 0.94 |
# | ValueBased3 | 3.85 | 0.66 | 6.00 | 0.31 | 16.38 | 0.94 |
# | ValueBased4 | 4.87 | 0.81 | 6.08 | 0.38 | 16.38 | 0.94 |
# 
# ### Results for a tournament with 3 value-based players and one 'random' player (24 games)
# 
# We see that the random player does not perform well.
# 
# Overall tournament results:
# 
# | Player | Avg Points | Std Points | Avg Victory Pts | Std Victory Pts | Avg Rounds | Std Rounds |
# |---|---:|---:|---:|---:|---:|---:|
# | RandomPlayer | 0.12 | 0.07 | 2.17 | 0.10 | 15.46 | 0.99 |
# | ValueBased2 | 6.02 | 0.66 | 6.42 | 0.33 | 15.46 | 0.99 |
# | ValueBased3 | 5.02 | 0.63 | 5.42 | 0.35 | 15.46 | 0.99 |
# | ValueBased4 | 5.83 | 0.70 | 6.33 | 0.34 | 15.46 | 0.99 |
# 
# ### Results for 4 equal (value based) players with 96 games in the tournament
# 
# We see the standard deviation reduced and the results closer to the expected average of 4.25 points.
# 
# Overall tournament results:
# 
# | Player | Avg Points | Std Points | Avg Victory Pts | Std Victory Pts | Avg Rounds | Std Rounds |
# |---|---:|---:|---:|---:|---:|---:|
# | ValueBased1 | 3.94 | 0.35 | 5.69 | 0.18 | 17.10 | 0.74 |
# | ValueBased2 | 4.14 | 0.39 | 5.59 | 0.19 | 17.10 | 0.74 |
# | ValueBased3 | 4.59 | 0.38 | 5.94 | 0.18 | 17.10 | 0.74 |
# | ValueBased4 | 4.33 | 0.37 | 5.77 | 0.19 | 17.10 | 0.74 |
# 

# %%
print("Running a tournament with 4 value-based players which should perform equally well.")
players = [
        pc.ValueBasedCatanPlayer(pc.default_structure, name="ValueBased1"),
        pc.ValueBasedCatanPlayer(pc.default_structure, name="ValueBased2"),
        pc.ValueBasedCatanPlayer(pc.default_structure, name="ValueBased3"),
        pc.ValueBasedCatanPlayer(pc.default_structure, name="ValueBased4")
    ]

no_games_in_tournament = 24

tournament = pc.Tournament(no_games_in_tournament=no_games_in_tournament,
                           max_rounds_per_game=50,
                           victory_points_to_win=8,
                           verbose=False)
overall_tournament_points, overall_victory_points, overall_rounds = tournament.tournament( players = players)
tournament.print_tournament_results(overall_tournament_points, overall_victory_points, overall_rounds, players)

# %%
print("Running a tournament with 3 value-based players and one 'random' player. \n" \
    "The random player is expected to perform worse.")
players = [
        pc.RandomCatanPlayer(pc.default_structure, name="RandomPlayer"),
        pc.ValueBasedCatanPlayer(pc.default_structure, name="ValueBased2"),
        pc.ValueBasedCatanPlayer(pc.default_structure, name="ValueBased3"),
        pc.ValueBasedCatanPlayer(pc.default_structure, name="ValueBased4")
    ]

no_games_in_tournament = 24

tournament = pc.Tournament(no_games_in_tournament=no_games_in_tournament,
                           max_rounds_per_game=50,
                           victory_points_to_win=8,
                           verbose=False)
overall_tournament_points, overall_victory_points, overall_rounds = tournament.tournament( players = players)
tournament.print_tournament_results(overall_tournament_points, overall_victory_points, overall_rounds, players)

# %%
print("If we increase the number of games the standard deviation should reduce.")
players = [
        pc.ValueBasedCatanPlayer(pc.default_structure, name="ValueBased1"),
        pc.ValueBasedCatanPlayer(pc.default_structure, name="ValueBased2"),
        pc.ValueBasedCatanPlayer(pc.default_structure, name="ValueBased3"),
        pc.ValueBasedCatanPlayer(pc.default_structure, name="ValueBased4")
    ]

no_games_in_tournament = 96

tournament = pc.Tournament(no_games_in_tournament=no_games_in_tournament,
                           max_rounds_per_game=50,
                           victory_points_to_win=8,
                           verbose=False)
overall_tournament_points, overall_victory_points, overall_rounds = tournament.tournament( players = players)
tournament.print_tournament_results(overall_tournament_points, overall_victory_points, overall_rounds, players)


