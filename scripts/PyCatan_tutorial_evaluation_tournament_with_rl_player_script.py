# %% [markdown]
# # PyCatan Tutorial 4

# %%
from pathlib import Path
import sys, os
from IPython import get_ipython

# Add the repository 'src' directory to sys.path by searching upward from the current working directory
repo_root = Path.cwd()
for _ in range(6):
    if (repo_root / 'src').exists():
        break
    if repo_root.parent == repo_root:
        break
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root / 'src'))

try:
    import Py_Catan_AI as pc
except Exception as e:
    raise ImportError(
        "Failed to import `Py_Catan_AI`. Make sure you are running this notebook from the repository root "
        "or that the project's `src/` folder is available on sys.path. I tried to add: " + str(repo_root / 'src')
    ) from e

import boto3
from keras.models import load_model

# IPython helper
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
# # PyCatan Tutorials: Evaluation tournaments for AI players
# 
# This is the PyCatan 'evaluation tournament' tutorial, the fourth from a series of 6. This tutorial shows how we can evaluate the performance of a trained player. We do this by running a tournament against the value based ('heuristic') players. In the tournament per game 17 points are awarded based on the final ranking (10 for winner, 5 for number two, 2 for number 3). Note that these are not the 'victory points' you get for building villages and towns, these are points you get over the course of the tournament for winning games. If player share a ranking they also share the points. So,we know that if players have equal performance they will score on average 4.25 points per game. 
# 
# The evaluation tournament we run with on `RLModelBasedCatanPlayer`. This player uses three models. One for decision making during the setup phase, one for decision making during the gameplay phase and one for answering to trade requests from other player. When we create a player instance it uses three default models (which are models that are trained with reinforcement learning). We can also setup the player with initial models that come from bootstrapping (these are models that learned to imitate the heuristic player). With the evaluation tournament we can compare their performance.
# 
# As discussed in an earlier tutorial we vary the order of the players during a tournament, to avoid that a player gets an advantage by being first. We also know that by playing more games we can reduce the variation, and reduce the effect of luck on the players performance.
# 
# ### The tutorials
# 
# - Running a game, including generating a game visualization/video with players speaking in character
# - Running a tournament and collecting data
# - Data generation for bootstrapping a model
# - Model training for bootstrapping reinforcement learning
# - Reinforcement learning to train a model
# - Running an evaluation tournament to check the performance of a model
# 
# ### Directory structure
# 
# <pre>
# PyCatan/
# ├── docs/  # contains tutorials in Jupyter Notebook format
# ├── scripts/  # contains runnable helper scripts
# ├── bootstrap/  # contains data and models used for bootstrapping RL learning
# └── src/
#     └── Py_Catan_AI/
#         ├── models/   # contains the 'default' models used when creating a new player instance
#         ├── data/     # contains some default data
#         └── visuals/   # contains some images used for visualization
# </pre>
# 
# ## Results
# 
# In this tutorial we use the default models that have been optimized with reinforcement learning. They perform (significantly) better than the heuristic models we used in the earlier tutorials. Below you also see a table with results from bootstrapped models (learned to imitate the heuristic player). We see that the performance of these models is (significantly) worse than the heuristic players.
# 
# Overall tournament results for 24 games:
# 
# | Player | Avg Points | Std Points | Avg Victory Pts | Std Victory Pts | Avg Rounds | Std Rounds |
# |:-------|-----------:|-----------:|----------------:|----------------:|-----------:|-----------:|
# | RL Model Player | 5.55 | 0.86 | 6.62 | 0.30 | 21.12 | 1.99 |
# | Value Based 1 | 2.78 | 0.49 | 5.58 | 0.31 | 21.12 | 1.99 |
# | Value Based 2 | 5.28 | 0.74 | 6.38 | 0.31 | 21.12 | 1.99 |
# | Value Based 3 | 3.39 | 0.69 | 5.75 | 0.35 | 21.12 | 1.99 |
# 
# Overall tournament results for 96 games:
# 
# | Player          | Avg Points | Std Points | Avg Victory Pts | Std Victory Pts | Avg Rounds | Std Rounds |
# |-----------------|-----------:|-----------:|----------------:|----------------:|-----------:|-----------:|
# | RL Model Player | 5.68       | 0.38       | 6.35            | 0.18            | 17.77      | 0.92       |
# | Hall 9000       | 3.86       | 0.37       | 5.75            | 0.16            | 17.77      | 0.92       |
# | Miss Minutes    | 3.80       | 0.38       | 5.66            | 0.18            | 17.77      | 0.92       |
# | C-3PO           | 3.67       | 0.32       | 5.57            | 0.16            | 17.77      | 0.92       |
# 
# 
# Overall tournament results for 400 games:
# 
# | Player           | Avg Points | Std Points | Avg Victory Pts | Std Victory Pts | Avg Rounds | Std Rounds |
# |------------------|-----------:|-----------:|----------------:|----------------:|-----------:|-----------:|
# | RL Model Player  | 5.47       | 0.20       | 6.43            | 0.09            | 18.98      | 0.44       |
# | Hall 9000        | 3.77       | 0.17       | 5.68            | 0.08            | 18.98      | 0.44       |
# | Miss Minutes     | 4.07       | 0.18       | 5.78            | 0.09            | 18.98      | 0.44       |
# | C-3PO            | 3.69       | 0.17       | 5.61            | 0.08            | 18.98      | 0.44       |
# 
# For reference, we can also run the tournament with the models we use at the start of the reinforcement learning (the models we use for 'bootstapping', trained to imitate the heuristic player without further optimization). These models perform worse.
# 
# Overall tournament results for 24 games (bootstrap models):
# 
# | Player | Avg Points | Std Points | Avg Victory Pts | Std Victory Pts | Avg Rounds | Std Rounds |
# |:-------|-----------:|-----------:|----------------:|----------------:|-----------:|-----------:|
# | RL Model Player | 3.24 | 0.77 | 5.04 | 0.38 | 16.54 | 1.53 |
# | Value Based 1 | 4.28 | 0.67 | 5.62 | 0.36 | 16.54 | 1.53 |
# | Value Based 2 | 5.22 | 0.86 | 5.92 | 0.46 | 16.54 | 1.53 |
# | Value Based 3 | 4.26 | 0.59 | 5.75 | 0.31 | 16.54 | 1.53 |
# 
# 
# Overall tournament results for 400 games (additional run):
# 
# | Player | Avg Points | Std Points | Avg Victory Pts | Std Victory Pts | Avg Rounds | Std Rounds |
# |:-------|-----------:|-----------:|----------------:|----------------:|-----------:|-----------:|
# | RL Model Player | 3.86 | 0.18 | 5.57 | 0.09 | 16.89 | 0.34 |
# | Value Based 1 | 4.41 | 0.19 | 5.80 | 0.09 | 16.89 | 0.34 |
# | Value Based 2 | 4.59 | 0.19 | 5.95 | 0.09 | 16.89 | 0.34 |
# | Value Based 3 | 4.14 | 0.18 | 5.71 | 0.09 | 16.89 | 0.34 |
# 
# 
# 

# %% [markdown]
# ## Parameters

# %%
print("For this tutorial we hard code some parameters. If you want to use command line" \
    "arguments in the script remove this block")

args.phase_to_train = 'None'  # 'TRADE' or 'GAMEPLAY' or 'SETUP' or 'None'
args.starting_model_name_trade = "./src/Py_Catan_AI/models/rl_decision_default_trade_model.keras"
args.starting_model_name_gameplay = "./src/Py_Catan_AI/models/rl_decision_default_gameplay_model.keras"
args.starting_model_name_setup = "./src/Py_Catan_AI/models/rl_decision_default_setup_model.keras"
args.number_of_games_in_evaluation_tournament = 24
args.filename_for_tournament_log = "tournament_log_identifier.csv"

print("Using hard coded parameters.")

# %%
def run_and_log_tournament(args):
    # === Setup player with rl model ===
    rl_player = pc.RLModelBasedCatanPlayer(
        name="RL Model Player",
        persona="A Catan player that plays based on a trained reinforcement learning model"
    )
    
    # ===  Load models ===
    # For the right phase load the model that is being trained from args.model_name_on_local
    # For others we use the starting models provided in args. we assume the models are already
    # on the local drive.
    #
    # For evaluation we set exploration to False
    if args.phase_to_train == 'TRADE':
        # load working model for trade phase
        rl_model_trade = pc.RLDecisionModel(structure=pc.default_structure)
        rl_model_trade.model = load_model(args.model_name_on_local, safe_mode=False, compile=False)
        rl_player.rl_model_for_trade_response = rl_model_trade
        rl_player.rl_model_for_trade_response.explore = False
        # load starting model for gameplay
        model_gameplay = pc.RLDecisionModel(structure=pc.default_structure)
        model_gameplay.model = load_model(args.starting_model_name_gameplay, safe_mode=False, compile=False)
        rl_player.rl_model_for_gameplay_phase = model_gameplay
        rl_player.rl_model_for_gameplay_phase.explore = False
        # load starting model for setup
        model_setup = pc.RLDecisionModel(structure=pc.default_structure)
        model_setup.model = load_model(args.starting_model_name_setup, safe_mode=False, compile=False)
        rl_player.rl_model_for_setup_phase = model_setup
        rl_player.rl_model_for_setup_phase.explore = False
    elif args.phase_to_train == 'GAMEPLAY':
        # load working model for gameplay phase
        rl_model_gameplay = pc.RLDecisionModel(structure=pc.default_structure)
        rl_model_gameplay.model = load_model(args.model_name_on_local, safe_mode=False, compile=False)
        rl_player.rl_model_for_gameplay_phase = rl_model_gameplay
        rl_player.rl_model_for_gameplay_phase.explore = False
        # load starting model for trade
        rl_model_trade = pc.RLDecisionModel(structure=pc.default_structure)
        rl_model_trade.model = load_model(args.starting_model_name_trade, safe_mode=False, compile=False)
        rl_player.rl_model_for_trade_response = rl_model_trade
        rl_player.rl_model_for_trade_response.explore = False
        # load starting model for setup
        rl_model_setup = pc.RLDecisionModel(structure=pc.default_structure)
        rl_model_setup.model = load_model(args.starting_model_name_setup, safe_mode=False, compile=False)
        rl_player.rl_model_for_setup_phase = rl_model_setup
        rl_player.rl_model_for_setup_phase.explore = False
    elif args.phase_to_train == 'SETUP':
        # load working model for setup phase
        rl_model_setup = pc.RLDecisionModel(structure=pc.default_structure)
        rl_model_setup.model = load_model(args.model_name_on_local, safe_mode=False, compile=False)
        rl_player.rl_model_for_setup_phase = rl_model_setup
        rl_player.rl_model_for_setup_phase.explore = False
        # load starting model for trade
        rl_model_trade = pc.RLDecisionModel(structure=pc.default_structure)
        rl_model_trade.model = load_model(args.starting_model_name_trade, safe_mode=False, compile=False)
        rl_player.rl_model_for_trade_response = rl_model_trade
        rl_player.rl_model_for_trade_response.explore = False
        # load starting model for gameplay
        model_gameplay = pc.RLDecisionModel(structure=pc.default_structure)
        model_gameplay.model = load_model(args.starting_model_name_gameplay, safe_mode=False, compile=False)
        rl_player.rl_model_for_gameplay_phase = model_gameplay
        rl_player.rl_model_for_gameplay_phase.explore = False
    else:
        # load starting model for setup
        model_setup = pc.RLDecisionModel(structure=pc.default_structure)
        model_setup.model = load_model(args.starting_model_name_setup, safe_mode=False, compile=False)
        rl_player.rl_model_for_setup_phase = model_setup
        rl_player.rl_model_for_setup_phase.explore = False
        # load starting model for trade
        rl_model_trade = pc.RLDecisionModel(structure=pc.default_structure)
        rl_model_trade.model = load_model(args.starting_model_name_trade, safe_mode=False, compile=False)
        rl_player.rl_model_for_trade_response = rl_model_trade
        rl_player.rl_model_for_trade_response.explore = False
        # load starting model for gameplay
        model_gameplay = pc.RLDecisionModel(structure=pc.default_structure)
        model_gameplay.model = load_model(args.starting_model_name_gameplay, safe_mode=False, compile=False)
        rl_player.rl_model_for_gameplay_phase = model_gameplay
        rl_player.rl_model_for_gameplay_phase.explore = False

    
    # === Set up total player list ===
    players = pc.default_players.copy()
    players[0] = rl_player
    for i in range(1,4):
        players[i].name = f"Value Based {i}"

    # === Run tournament and collect logs ===
    tr = pc.Tournament(no_games_in_tournament=args.number_of_games_in_evaluation_tournament)
    tr.verbose = False
    overall_tournament_points, overall_victory_points, overall_rounds = tr.tournament_rl_training_data_generation(players)   
    tr.print_tournament_results(overall_tournament_points, overall_victory_points, overall_rounds, players)

    # === Log tournament results to csv ===
    df_tournament_log = tr.log_tournament_results_in_dataframe(args.training_round,overall_tournament_points, overall_victory_points, overall_rounds, players)
    save_path = args.filename_for_tournament_log.replace("identifier", str(args.training_round))
    df_tournament_log.to_csv(save_path, index=False)

    # === Log results to S3 (optional) ===
    if args.s3_bucket_name is not None:
        s3 = boto3.client("s3")
        s3.upload_file(save_path, args.s3_bucket_name, f"results/{save_path}")
        print(f"✅ Uploaded tournament log to s3://{args.s3_bucket_name}/results/{save_path}")

    # === Provide feedback ===
    if args.phase_to_train in ['TRADE', 'GAMEPLAY', 'SETUP']:
        print(f"=== Finished evaluation tournament round {args.training_round} with model {args.model_name_on_local} on {args.number_of_games_in_evaluation_tournament} tournaments===")
    else:
        print(f"=== Finished evaluation tournament round {args.training_round} with starting models on {args.number_of_games_in_evaluation_tournament} tournaments===")
    return

# %%
run_and_log_tournament(args)

# %%
# We can also use the models generated from imitation learning. They should perform similar, or slightly worse
# compared to the value based players.

args.phase_to_train = 'None'  # 'TRADE' or 'GAMEPLAY' or 'SETUP' or 'None'
args.starting_model_name_trade = "./src/Py_Catan_AI/models/rl_decision_model_bootstrap_trade.keras"
args.starting_model_name_gameplay = "./src/Py_Catan_AI/models/rl_decision_model_bootstrap_gameplay.keras"
args.starting_model_name_setup = "./src/Py_Catan_AI/models/rl_decision_model_bootstrap_setup.keras"
args.number_of_games_in_evaluation_tournament = 24
args.filename_for_tournament_log = "tournament_log_identifier.csv"

# %%
run_and_log_tournament(args)


