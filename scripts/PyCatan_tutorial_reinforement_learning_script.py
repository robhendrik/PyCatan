# %% [markdown]
# # PyCatan Tutorial 6

# %%
from pathlib import Path
import sys, os
import pickle
import numpy as np
import pandas as pd
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
    raise ImportError("Failed to import `Py_Catan_AI`. Make sure you are running this notebook from the repository root or that the project's `src/` folder is available on sys.path. I tried to add: " + str(repo_root / 'src')) from e

# Standard libraries used in the tutorials
import tensorflow as tf
import boto3
import argparse
import random
from keras.models import load_model
import time
import scripts as sc

# IPython helper
ip = get_ipython()


# %%
# # Set working directory to repo root
# os.chdir(pc.get_repo_root(target_name="PyCatan"))
# s3 = boto3.client('s3')
# s3.upload_file("src/Py_Catan_AI/models/rl_decision_default_trade_model.keras", 
#                "pycatanbucket", 
#                "bootstrap/rl_decision_default_trade_model.keras")

# "PyCatan/src/Py_Catan_AI/models/rl_decision_model_bootstrap_setup.keras"
# "PyCatan/src/Py_Catan_AI/models/rl_decision_model_bootstrap_trade.keras"
# "PyCatan/src/Py_Catan_AI/models/rl_decision_model_bootstrap_gameplay.keras"
# "PyCatan/src/Py_Catan_AI/models/rl_decision_default_setup_model.keras"
# "PyCatan/src/Py_Catan_AI/models/rl_decision_default_gameplay_model.keras"
# "PyCatan/src/Py_Catan_AI/models/rl_decision_default_trade_model.keras"


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
# # PyCatan Tutorial: Reinforcement Learning
# 
# This notebook demonstrates how to train and evaluate reinforcement-learning (RL) players for the three Catan phases: SETUP, GAMEPLAY and TRADE. It includes: downloading starting models, generating training data via self-play, preparing datasets, training with PPO, and running evaluation tournaments.
# 
# Follow the cells in order. The notebook is designed to run from the repository root.
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

# %% [markdown]
# ## Parameters

# %%
args.phase_to_train = 'TRADE' # 'TRADE' or 'GAMEPLAY' or 'SETUP'
args.training_rounds = 50 # for training
args.training_round = 0 # first round, if this is 10 and training rounds is 20 we run rounds from 10 to 29
args.tournaments_per_round = 1 # 4 # for generating training data
args.games_per_tournament = 4 #24 # for generating training data

# Evaluation tournament settings
args.number_of_games_in_evaluation_tournament = 96 # 24 # for evaluating performance per training round
args.evaluation_tournament_interval = 5 # run evaluation tournament every N rounds
args.filename_for_tournament_log = "tutorial_tournament_log_identifier.csv" # identifier will be replaced with training round number # data from evaluation tournaments

# training parameters
args.entropy_coef = 0.12
args.value_coef = 0.5
args.learning_rate = 1e-4 # 5e-6
args.clipnorm = 0.5
args.epsilon = 1e-7
args.clip_ratio = 0.2
args.epochs = 20
args.batch_size = 128
args.target_kl = 0.25
args.lr_backoff = 0.8
args.bc_beta_HARDCODED = 0.00  # try 0.05–0.1

# Discount factor
args.gamma = 0.99 # discount factor for future rewards

# initial models
args.s3_bucket_name = "pycatanbucket"
# We have these models as part of the package: 
# "PyCatan/src/Py_Catan_AI/models/rl_decision_model_bootstrap_setup.keras"
# "PyCatan/src/Py_Catan_AI/models/rl_decision_model_bootstrap_trade.keras"
# "PyCatan/src/Py_Catan_AI/models/rl_decision_model_bootstrap_gameplay.keras"
# "PyCatan/src/Py_Catan_AI/models/rl_decision_default_setup_model.keras"
# "PyCatan/src/Py_Catan_AI/models/rl_decision_default_gameplay_model.keras"
# "PyCatan/src/Py_Catan_AI/models/rl_decision_default_trade_model.keras"
args.starting_model_name_gameplay = "bootstrap/rl_decision_model_bootstrap_gameplay.keras"
args.starting_model_name_setup = "bootstrap/rl_decision_default_setup_model.keras"
args.starting_model_name_trade = "bootstrap/rl_decision_default_trade_model.keras"

# working model (for phase to be trained)
# model will be stored on S3 as models/working_model_3_identifier.keras, identifier will be replaced with training round number
args.filename_for_storing_model_on_s3 = "models/working_model_tutorial_identifier.keras"
args.model_name_on_local = "working_model.keras"

# initial data for stabilizing training, download pre-generated data from S3
args.number_of_files_to_download = 5 # 12
args.number_of_stable_datasets_to_mix_in = 3
args.number_of_previous_datasets_to_mix_in = 3
# logs will be stored on S3 as data/rl_log_tutorial_ppo_round_0.pkl, data/rl_log_tutorial_ppo_round_1.pkl, ...
args.filename_for_storing_rl_log_on_s3 = "rl_log_tutorial_ppo_round_identifier.pkl" # identifier will be replaced with training round number
# we will select args.number_of_files_to_download files from the following full list
# We have data stored in this directory with meta_indicator 0 or 1 and tournament_indicator from 0 to 49
# "PyCatan/src/Py_Catan_AI/data/bootstrap_meta{meta_indicator}_tournament{tournament_indicator}_rl_log.pkl"
args.full_list_of_filenames_on_s3 = [f"bootstrap/bootstrap_meta{meta_indicator}_tournament{tournament_indicator}_rl_log.pkl" for meta_indicator in range(2) for tournament_indicator in range(50)]

# parallelization settings, only relevant if exploiting multi-core setup
args.number_of_workers = 8
args.games_per_worker = 24

# Varying or fixed permutation pf player order
args.fixed_player_order = False # Random is more representant, fixed will give less variance and show progress more quickly


# %% [markdown]
# ## Download the starting models from S3
# 
# This step downloads the baseline models used by the RL player. Two models are typically kept fixed while one model is trained for the selected phase. This 'working model' is stored locally and periodically uploaded to S3 as a checkpoint.
# 
# Notes:
# - The helper function below downloads the configured starting models and also creates the local working copy used for training.
# - Ensure `args.s3_bucket_name` and the starting-model names in `args` are set correctly before running the download cell.
# - If a local working-model file already exists it will be removed and replaced by the downloaded copy (with a retry loop to handle transient filesystem locks).

# %%
def download_starting_models_from_s3_to_local(args) -> None:    
    """Download starting models from S3 to local. Also generate a working model on local for training.

    Args:
        args (any): arguments containing the starting model names.
    """
    # == Set S3 settings ===
    s3 = boto3.client('s3')
    bucket_name = args.s3_bucket_name

    # === Load initial models from S3 ===
    s3.download_file(bucket_name, args.starting_model_name_setup, args.starting_model_name_setup)
    s3.download_file(bucket_name, args.starting_model_name_trade, args.starting_model_name_trade)
    s3.download_file(bucket_name, args.starting_model_name_gameplay, args.starting_model_name_gameplay)
    print(f"Downloaded starting models from s3://{bucket_name}/ to local. Setup: {args.starting_model_name_setup}, Trade: {args.starting_model_name_trade}, Gameplay: {args.starting_model_name_gameplay}")
    # === Load model from S3 for training ===
    if args.phase_to_train == 'TRADE':
        load_path = args.starting_model_name_trade
    elif args.phase_to_train == 'GAMEPLAY':
        load_path = args.starting_model_name_gameplay
    elif args.phase_to_train == 'SETUP':
        load_path = args.starting_model_name_setup
    else:
        raise ValueError(f"Unknown phase_to_train: {args.phase_to_train}")
    if os.path.exists(args.model_name_on_local):
        while True:
            try:
                os.remove(args.model_name_on_local)
                break
            except Exception as e:
                print(f"Error removing existing file {args.model_name_on_local}: {e}. Retrying in 30 seconds.")
                time.sleep(30)

    s3.download_file(bucket_name, f"{load_path}", args.model_name_on_local)

    # === Feedback ===
    print(f"Downloaded starting model from s3://{bucket_name}/{load_path} to {args.model_name_on_local} as working model to be trained.")

    return 

# %% [markdown]
# ## Data generation
# 
# This function is called to run tournaments to generate data. This data is then later evaluated and used to update the model. The function is written such that it can be used for a multi-core setup (vary the worker index in that case). The generated logs are added to the list 'list_of_rl_logs'.

# %%
def worker_job( worker_assignment: list[tuple] = [(0,24)],
                worker_idx: int = 0,
                rl_player = pc.RLModelBasedCatanPlayer(),
                list_of_rl_logs: list = [],
                args: any = None,
                ):
    """ For every tuple (start, stop) in worker_assignment, run a tournament with games from start to stop-1.
    Collect all logs in list_of_rl_logs. To generate on log concatenated dataframe after all workers are done, use:
    all_logs_df = pd.concat(list_of_rl_logs, ignore_index=True).

    We use the worker setup to exploit multi-core setups, where each worker can run on a separate core. If this is not needed, 
    just run one worker_job with index 0 (default) and worker_assignment [(0, number_of_games)].

    The rl_player is assumed to be already setup with the right models for the right phases. It will play against default players.
    Exploration noise is enabled during training data generation for this player.

    The meta_indicator column in the logs is set to 1234 for all logs generated here, so they can be identified later.

    Args:
        worker_assignment (list, optional): List of tuples (start, stop) indicating the games to be played by this worker. Defaults to [(0,24)].
        worker_idx (int, optional): Index of the worker. Defaults to 0.
        rl_player (RLModelBasedCatanPlayer, optional): The RL player to be used in the tournament. Defaults to RLModelBasedCatanPlayer().
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
        list_of_rl_logs (list, optional): List to collect the logs. Defaults to [].
        fixed_player_order (bool, optional): Whether to use a fixed player order. Defaults to False.
    
    Returns:
        None: Logs are collected in list_of_rl_logs as a list of dataframes.
    """
    fixed_player_order = args.fixed_player_order
    for tournament_index, (start, stop) in enumerate(worker_assignment):
        print(f"Worker {worker_idx} assigned games {start}-{stop-1}")
        players = pc.default_players.copy()
        players[0] = rl_player

        assert 'Setup' in rl_player.rl_model_for_setup_phase.name, "Setup phase model not set correctly"
        assert 'Trade' in rl_player.rl_model_for_trade_response.name, "Trade response model not set correctly"
        assert 'Gameplay' in rl_player.rl_model_for_gameplay_phase.name, "Gameplay phase model not set correctly"
        
        if args.phase_to_train == 'TRADE':
            assert 'Working' in rl_player.rl_model_for_trade_response.name, "Trade response model"
            rl_player.rl_model_for_setup_phase.explore = False  # Disable exploration noise during setup phase
            rl_player.rl_model_for_gameplay_phase.explore = False  # Disable exploration noise during gameplay phase
            rl_player.rl_model_for_trade_response.explore = True  # Enable exploration noise during training data generation
        elif args.phase_to_train == 'GAMEPLAY':
            assert 'Working' in rl_player.rl_model_for_gameplay_phase.name, "Gameplay phase model not set correctly for TRADE training"
            rl_player.rl_model_for_setup_phase.explore = False  # Disable exploration noise during setup phase
            rl_player.rl_model_for_gameplay_phase.explore = True  # Enable exploration noise during training data generation
            rl_player.rl_model_for_trade_response.explore = False  # Disable exploration noise during trade response
        elif args.phase_to_train == 'SETUP':
            assert 'Working' in rl_player.rl_model_for_setup_phase.name, "Setup phase model not set correctly for SETUP training"
            rl_player.rl_model_for_setup_phase.explore = True  # Enable exploration noise during training data generation
            rl_player.rl_model_for_gameplay_phase.explore = False  # Disable exploration noise during gameplay phase
            rl_player.rl_model_for_trade_response.explore = False  # Enable exploration noise during training data generation
        


        # === Run tournament and collect logs ===
        t = pc.Tournament()
        t.verbose = False
        rl_log = t.tournament_rl_training_data_generation( players = players, 
                                                        gamma=None,
                                                        start_game_number=start,
                                                        stop_game_number=stop,
                                                        fixed_player_order=fixed_player_order,
                                                        output_type='logs_only')
        for log in rl_log:
            log["tournament_indicator"] = tournament_index
            log["meta_indicator"] = 1234
        list_of_rl_logs.extend(rl_log)

        # === Feedback ===
        print(f"Worker {worker_idx} done.")

    return

# %%
def  main(args):
    # === Log the parameters ===
    pc.print_args(args)
    
    # === Set S3 settings ===
    s3 = boto3.client("s3")
    bucket_name = args.s3_bucket_name

    # === Download starting models from S3 to local and create a working model on local ===
    download_starting_models_from_s3_to_local(args)
        
    # === Create initial list of data files and download them from S3
    full_list_of_filenames = args.full_list_of_filenames_on_s3
    
    # === Randomly select files to download
    s3_log_paths = np.random.choice(full_list_of_filenames, size=args.number_of_files_to_download, replace=False).tolist()
    old_log_paths = []
    for filename in s3_log_paths:
        save_path = filename.split("/")[-1]
        try:
            s3.download_file(bucket_name, filename, save_path) 
            old_log_paths.append(save_path)
        except Exception as e:
            print(f"Error downloading {filename} to hgh.pkl: {e}")
    print(f"Downloaded {len(old_log_paths)} log files from S3.")

    # === Create rl player and load models ===
    # This player will be used to generate training data and will be updated
    # The player has to be set to explore. The only model that is trained
    # is the gameplay model, the others are kept fixed.
    rl_player = pc.RLModelBasedCatanPlayer(
        name="RL Model Player",
        persona="A Catan player that plays based on a trained reinforcement learning model"
    )
    # === load models for setup ===
    if args.phase_to_train != 'SETUP':
        rl_player.rl_model_for_setup_phase.model = tf.keras.models.load_model(
                args.starting_model_name_setup,
                safe_mode=False,
                compile=False
                
            )
        rl_player.rl_model_for_setup_phase.name = "Setup Phase Model"
        rl_player.rl_model_for_setup_phase.explore = False
    # === load models for trade ===
    if args.phase_to_train != 'TRADE':
        rl_player.rl_model_for_trade_response.model = tf.keras.models.load_model(
                args.starting_model_name_trade,
                safe_mode=False,
                compile=False
            )
        rl_player.rl_model_for_trade_response.name = "Trade Response Model"
        rl_player.rl_model_for_trade_response.explore = False
    # === load model for gameplay ===
    if args.phase_to_train != 'GAMEPLAY':
        rl_player.rl_model_for_gameplay_phase.model = tf.keras.models.load_model(
                args.starting_model_name_gameplay,
                safe_mode=False,
                compile=False
            )
        rl_player.rl_model_for_gameplay_phase.name = "Gameplay Phase Model"
        rl_player.rl_model_for_gameplay_phase.explore = False  

    if args.phase_to_train == 'TRADE':
        rl_player.rl_model_for_trade_response.model = tf.keras.models.load_model(
            args.model_name_on_local,
            safe_mode=False,
            compile=False
        )
        rl_player.rl_model_for_trade_response.name = "Trade Response Working Model"
        rl_player.rl_model_for_trade_response.explore = True
        print("Loaded initial trade model into RL player.")
    elif args.phase_to_train == 'GAMEPLAY':
        rl_player.rl_model_for_gameplay_phase.model = tf.keras.models.load_model(
            args.model_name_on_local,
            safe_mode=False,
            compile=False
        )
        rl_player.rl_model_for_gameplay_phase.name = "Gameplay Phase Working Model"
        rl_player.rl_model_for_gameplay_phase.explore = True
        print("Loaded initial gameplay model into RL player.")
    elif args.phase_to_train == 'SETUP':
        rl_player.rl_model_for_setup_phase.model = tf.keras.models.load_model(
            args.model_name_on_local,
            safe_mode=False,
            compile=False
        )
        rl_player.rl_model_for_setup_phase.name = "Setup Working Model"
        rl_player.rl_model_for_setup_phase.explore = True
        print("Loaded initial setup model into RL player.")
    else:
        raise ValueError("PHASE must be 'TRADE', 'GAMEPLAY' or 'SETUP'")
    print("Loaded model into RL player.")

    # === Main training loop ===
    # We keep a record of each round's model on S3 for later evaluation
    s3_model_filename = args.filename_for_storing_model_on_s3.replace("identifier", str(args.training_round))
    s3.upload_file(args.model_name_on_local, bucket_name, s3_model_filename)
    print(f"Loaded starting model to s3://{bucket_name}/{s3_model_filename}")

    # list to store generated log filenames to use across training rounds
    list_of_all_generated_logs = []

    # === Training loop ===
    for training_round in range(args.training_round, args.training_round + args.training_rounds):
        args.training_round = training_round
        
        # === Evaluate performance before training and save the log to a file with an identifier
        if args.number_of_games_in_evaluation_tournament > 0 and training_round % args.evaluation_tournament_interval == args.evaluation_tournament_interval-1:
            print(f"=== Starting evaluation tournament round {training_round} with model {args.model_name_on_local} on {args.number_of_games_in_evaluation_tournament} games for phase {args.phase_to_train} ===")
            sc.run_and_log_tournament(args)
        else:
            if args.number_of_games_in_evaluation_tournament == 0:
                print(f"=== Skipping evaluation tournament round {training_round} as number_of_games_in_evaluation_tournament is set to 0 ===")
            else:
                print(f"=== Skipping evaluation tournament round {training_round} as per interval {args.evaluation_tournament_interval} setting ===")
        
        if args.games_per_tournament <= 0 or args.tournaments_per_round <= 0:
            print("Skipping data generation as games_per_tournament or tournaments_per_round is set to 0.")
            added_old_log_paths = random.sample(old_log_paths, args.number_of_stable_datasets_to_mix_in) # the original files contain 50 games, so for 300 games we add 6 files
            training_log_paths = added_old_log_paths
            print(f"Using {len(training_log_paths)} log files for training: {', '.join(training_log_paths)}")
        else:
            # === Run round of tournament to generate data
            list_of_rl_logs = []
            assignments = [(0,args.games_per_tournament) for _ in range(args.tournaments_per_round)]  
            worker_job(worker_assignment = assignments, 
                    worker_idx=0, 
                    rl_player=rl_player, 
                    list_of_rl_logs=list_of_rl_logs,
                    args=args
                    )
            new_rl_log = pd.concat(list_of_rl_logs, ignore_index=True)
            new_rl_log['meta_indicator'] = 2234 + training_round
            
            # === Perform sanity checks on new log
            assert 'meta_indicator' in new_rl_log.columns, "meta_indicator column missing in new_rl_log"
            assert (new_rl_log['meta_indicator'] == 2234+training_round).all(), "meta_indicator column in new_rl_log must be 2234 for all entries"
            assert (new_rl_log['player'] == rl_player.name).all(), "player column in new_rl_log must be the rl_player name for all entries"
            assert 'game_number' not in new_rl_log.columns, "game_number column should not be in new_rl_log"
            assert 'game_indicator' in new_rl_log.columns, "game_indicator column missing in new_rl_log"
            print(f"{len(assignments)} tournaments completed with in total {len(new_rl_log)} games.")
            
            # === Store new logs as pickle files
            filename = args.filename_for_storing_rl_log_on_s3.replace("identifier", str(training_round))
            with open(filename, "wb") as f:
                pickle.dump(new_rl_log, f)   # save raw log with policy_probs
            s3.upload_file(filename, bucket_name, f"data/{filename}")
            print(f"✅ Uploaded new rl log to s3://{bucket_name}/data/{filename}")
            list_of_all_generated_logs.append(filename)

            # === Select data from the original data and/or previous rounds for stabilization ===
            logs_from_previous_rounds = list_of_all_generated_logs[:-1] # all previously generated logs
            logs_from_previous_rounds = logs_from_previous_rounds[max(0,training_round-args.number_of_previous_datasets_to_mix_in):] # only keep last N
            added_old_log_paths = random.sample(old_log_paths, max(0,args.number_of_stable_datasets_to_mix_in - training_round)) # the original files contain 50 games, so for 300 games we add 6 files
            
            # === Create dataset for training, from new generated data (with current model version) and some old data
            training_log_paths = [filename] + added_old_log_paths + logs_from_previous_rounds
            print(f"Using {len(training_log_paths)} log files for training: {', '.join(training_log_paths)}")
            
        # === Load all data files and concatenate them to single log
        list_of_logs = []
        for log_path in training_log_paths:
            with open(log_path, "rb") as f:
                rl_log = pickle.load(f)
                if not ("old_action_prob" in rl_log.columns and "policy_probs" in rl_log.columns):
                    print(f"⚠️ Warning: log file {log_path} is missing 'old_action_prob' or 'policy_probs' column.")
                if 'tournamen_indicator' in rl_log.columns:
                    rl_log["tournament_indicator"] = rl_log["tournamen_indicator"]
                    rl_log = rl_log.drop(columns=["tournamen_indicator"])
                    print(f"Loaded {log_path} with {len(rl_log)} entries, column 'tournamen_indicator' renamed to 'tournament_indicator'.")
            list_of_logs.append(rl_log)
        rl_log = pd.concat(list_of_logs, ignore_index=True)
        
        # === Some sanity checks and renaming if needed ===
        if 'tournamen_indicator' in rl_log.columns:
            rl_log["tournament_indicator"] = rl_log["tournamen_indicator"]
            rl_log = rl_log.drop(columns=["tournamen_indicator"])
            print(f"Column 'tournamen_indicator' renamed to 'tournament_indicator'.")
        
        if "old_action_prob" in rl_log.columns and "policy_probs" in rl_log.columns:
            assert 1 == 1
            # pass
        else:
            if "policy_probs" in rl_log.columns:
                rl_log = rl_log.drop(columns=["policy_probs"])
            if "old_action_prob" in rl_log.columns:
                rl_log = rl_log.drop(columns=["old_action_prob"])

        # For now remove state_value from the dataframe to force recalculation based on current model.
        if "state_value" in rl_log.columns:
            rl_log = rl_log.drop(columns=["state_value"])

        # === Update per game future discounted rewards (based on gamma discount factor and earned victory points) ===
        unique_meta_ids = rl_log["meta_indicator"].unique()
        for meta_id in unique_meta_ids:
            unique_tournament_ids = rl_log[rl_log["meta_indicator"] == meta_id]["tournament_indicator"].unique()
            for tournament_id in unique_tournament_ids:
                unique_game_ids = rl_log[(rl_log["meta_indicator"] == meta_id) & (rl_log["tournament_indicator"] == tournament_id)]["game_indicator"].unique()
                for game_id in unique_game_ids:
                    log_mask = (rl_log["meta_indicator"] == meta_id) & (rl_log["tournament_indicator"] == tournament_id) & (rl_log["game_indicator"] == game_id)
                    updated = pc.finalize_rewards_on_single_game_rl_log(rl_log.loc[log_mask], gamma=args.gamma)
                    rl_log.loc[log_mask, ["delta_reward", "return", "advantage"]] = updated[["delta_reward", "return", "advantage"]]
        print("Updated rewards, returns, and advantages for all games in the combined log.")
        print(f"Before isolating {args.phase_to_train} the combined log has {len(rl_log)} entries from {len(training_log_paths)} files.")
        
        # === Keep only data for the phase we are training
        logs = pc.split_logs_by_phase(rl_log)
        rl_log = logs[args.phase_to_train.lower()]
        assert rl_log['phase'].eq(args.phase_to_train.lower()).all(),f"Log must contain only {args.phase_to_train} phase entries."
        print(f"After filtering for phase {args.phase_to_train}, {len(rl_log)} entries remain.")
        
        # === Filter out all entries where the mask only allows a single action (no decision made) ===
        rl_log = rl_log[rl_log["mask"].map(lambda m: np.count_nonzero(np.asarray(m) > 0.5) >= 2)].reset_index(drop=True)
        print(f"After removing entries with single-action masks, {len(rl_log)} entries remain.")

        if args.phase_to_train == 'TRADE':
            model_to_train = rl_player.rl_model_for_trade_response
        elif args.phase_to_train == 'GAMEPLAY':
            model_to_train = rl_player.rl_model_for_gameplay_phase
        elif args.phase_to_train == 'SETUP':
            model_to_train = rl_player.rl_model_for_setup_phase
        else:
            raise ValueError("PHASE must be 'TRADE', 'GAMEPLAY' or 'SETUP'")
        
        # === Prepare dataset for training with PPO ===
        dataset = pc.to_training_dataset_local(rl_log_phase = rl_log, 
                                                      structure=pc.default_structure, 
                                                      rl_model = model_to_train, 
                                                      normalize_adv=True)
        print(f"Created training dataset with {len(dataset["y_policy"])} entries from {len(rl_log)} log entries.")
        # dataset will have keys: x_inputs, y_policy, y_value, old_action_probs, adv
        # We use vector, mask (in x_inputs) from data as well as y_policy from action taken
        # y_value, adv and old_action_probs have been regenerated in this module to ensure consistency

        # === Train with PPO === !
        advantages = dataset["adv"]
        print(f"Advantage stats as PPO input: mean={np.mean(advantages):.4f}, std={np.std(advantages):.4f}, min={np.min(advantages):.4f}, max={np.max(advantages):.4f}")

        print(f"Training {args.phase_to_train} model with PPO...")
        ppo = pc.PPOTrainer(
            rl_model=model_to_train,
            entropy_coef=args.entropy_coef,  # 0.01
            value_coef=args.value_coef,  # 0.7
            learning_rate=args.learning_rate,  # 5e-6
            clipnorm=args.clipnorm,  # 0.5
            epsilon=args.epsilon,  # 1e-7
            clip_ratio=args.clip_ratio  # 0.1
        )
        ppo.train(dataset, epochs=args.epochs, batch_size=args.batch_size, target_kl=args.target_kl, lr_backoff=args.lr_backoff)

        # === Analyse and report back training effect ===
        eps = 1e-8

        # Fresh predictions from the (just-trained) model
        new_probs, new_values = model_to_train.predict_probabilities(dataset["x_inputs"], verbose=0)
        new_probs = np.asarray(new_probs, dtype=np.float64)

        # Unpack inputs
        x1, x2, x3, masks = dataset["x_inputs"]
        masks = np.asarray(masks, dtype=np.float64)

        def renorm_over_legal(p, m, eps=1e-8):
            """Clip, mask, and renormalize over legal actions (row-wise)."""
            p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
            pm = p * m
            z = pm.sum(axis=1, keepdims=True) + eps
            return pm / z

        # Behavior policy from the log (after exploration forcing), if available
        beh_probs = dataset.get("behavior_probs", None)
        have_behavior = beh_probs is not None
        if have_behavior:
            beh = renorm_over_legal(beh_probs, masks, eps)
        new = renorm_over_legal(new_probs, masks, eps)

        # Entropy (of NEW policy, legal-action)
        entropy = -np.mean(np.sum(new * np.log(new + eps), axis=1))

        # Value loss proxy (MSE to targets)
        y_true = np.asarray(dataset["y_value"], dtype=np.float32).reshape(-1, 1)
        y_pred = np.asarray(new_values, dtype=np.float32).reshape(-1, 1)
        value_mse = float(np.mean((y_true - y_pred) ** 2))

        # KL/JS/TV to behavior (only if we actually have the behavior distribution)
        kl_to_behavior = js = tv = float('nan')
        flip_rate = kl_two = kl_gt2 = flip_two = flip_gt2 = float('nan')
        if have_behavior:
            # Divergences to behavior
            kl_to_behavior = float(np.mean(np.sum(beh * (np.log(beh + eps) - np.log(new + eps)), axis=1)))
            m = 0.5 * (beh + new)
            js = float(np.mean(
                0.5 * np.sum(beh * (np.log(beh + eps) - np.log(m + eps)), axis=1) +
                0.5 * np.sum(new * (np.log(new + eps) - np.log(m + eps)), axis=1)
            ))
            tv = float(0.5 * np.mean(np.sum(np.abs(beh - new), axis=1)))

            # Decision flips overall and split by action count
            beh_argmax = np.argmax(beh, axis=1)
            new_argmax = np.argmax(new, axis=1)
            flip_rate = float(np.mean(beh_argmax != new_argmax))

            num_legal = masks.sum(axis=1)
            two_mask  = (num_legal == 2)
            gt2_mask  = (num_legal > 2)

            def safe_mean(vec, mask):
                return float(np.mean(vec[mask])) if np.any(mask) else float('nan')

            kl_rows = np.sum(beh * (np.log(beh + eps) - np.log(new + eps)), axis=1)
            flips   = (beh_argmax != new_argmax).astype(np.float64)

            kl_two  = safe_mean(kl_rows, two_mask)
            kl_gt2  = safe_mean(kl_rows, gt2_mask)
            flip_two = safe_mean(flips, two_mask)
            flip_gt2 = safe_mean(flips, gt2_mask)

        # PPO clip diagnostics on taken actions (works without behavior_probs)
        # Reconstruct taken actions from one-hots and use logged old_action_probs
        actions = np.argmax(dataset["y_policy"], axis=1)          # (N,)
        old_ap  = np.asarray(dataset["old_action_probs"], dtype=np.float64)  # (N,)
        new_ap  = new[np.arange(new.shape[0]), actions]
        ratio   = new_ap / np.clip(old_ap, eps, None)
        cr      = getattr(model_to_train, "clip_ratio", 0.1) if hasattr(model_to_train, "clip_ratio") else 0.1
        clipped_hi = float(np.mean(ratio > 1.0 + cr))
        clipped_lo = float(np.mean(ratio < 1.0 - cr))

        # Pretty print
        if have_behavior:
            print(
                f"[Round {training_round}] "
                f"KL_to_behavior={kl_to_behavior:.4f} | JS={js:.4f} | TV={tv:.4f} "
                f"| Entropy={entropy:.4f} | ValueMSE={value_mse:.4f} "
                f"| Flip={flip_rate:.3f} "
                f"| KL(two)={kl_two:.4f} KL(>2)={kl_gt2:.4f} "
                f"| Flip(two)={flip_two:.3f} Flip(>2)={flip_gt2:.3f} "
                f"| ClipFrac hi/lo={clipped_hi:.3f}/{clipped_lo:.3f}"
            )
        else:
            print(
                f"[Round {training_round}] "
                f"Entropy={entropy:.4f} | ValueMSE={value_mse:.4f} "
                f"(no behavior_probs → KL/JS/TV/Flip skipped) "
                f"| ClipFrac hi/lo={clipped_hi:.3f}/{clipped_lo:.3f}"
            )

        # === Sanity check ===
        assert 'Working' in model_to_train.name, "Model name not set correctly"
        if args.phase_to_train == 'TRADE':
            rl_player.rl_model_for_trade_response = model_to_train
            assert 'Trade' in model_to_train.name, "Model name not set correctly for TRADE training"
        elif args.phase_to_train == 'GAMEPLAY':
            rl_player.rl_model_for_gameplay_phase = model_to_train
            assert 'Gameplay' in model_to_train.name, "Model name not set correctly for GAMEPLAY training"
        elif args.phase_to_train == 'SETUP':
            rl_player.rl_model_for_setup_phase = model_to_train
            assert 'Setup' in model_to_train.name, "Model name not set correctly for SETUP training"
        else:
            raise ValueError("PHASE must be 'TRADE', 'GAMEPLAY' or 'SETUP'")
        
        # === Save checkpoint locally as working model (for restarting after interruption) ===
        model_to_train.model.save(args.model_name_on_local, include_optimizer=False)

        # === Upload model checkpoint to S3 with unique identifier for later evaluation ===
        s3_model_filename = args.filename_for_storing_model_on_s3.replace("identifier", str(training_round+1))
        s3.upload_file(args.model_name_on_local, bucket_name, s3_model_filename)
        print(f"✅ Uploaded model checkpoint from {args.model_name_on_local} to s3://{bucket_name}/{s3_model_filename}")


# %%
main(args)


# %%
print(f"=== Starting final evaluation tournament with model {args.model_name_on_local} on {args.number_of_games_in_evaluation_tournament} games for phase {args.phase_to_train} ===")
sc.run_and_log_tournament(args)


