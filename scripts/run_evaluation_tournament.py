import sys
import os
import argparse
import numpy as np
import pandas as pd
sys.path.append("./src")

from Py_Catan_AI.default_structure import default_players, default_structure
from Py_Catan_AI.model_based_catan_player import ModelBasedCatanPlayer
from Py_Catan_AI.rl_model_based_catan_player import RLModelBasedCatanPlayer
from Py_Catan_AI.rl_decision_model import RLDecisionModel
from Py_Catan_AI.rl_tournament import RLTournament, debug_dataset
from Py_Catan_AI.py_catan_tournament import Tournament
from Py_Catan_AI.ppo_trainer import PPOTrainer
from Py_Catan_AI.ppo_trainer import ppo_loss

import tensorflow as tf
import keras
import boto3

# python scripts/run_evaluation_tournament.py --model_identifier 12
s3 = boto3.client("s3")
bucket_name = "pycatanbucket"

def main(args):
    load_path = args.model_name
    s3.download_file(bucket_name, f"models/{load_path}", "model.keras")
    print(f"✅ Downloaded model checkpoint from s3://{bucket_name}/models/{load_path}")

    # --- Setup players ---
    original_player = ModelBasedCatanPlayer(name="original", persona="Original")
    rl_player = RLModelBasedCatanPlayer(
        name="RL Model Player",
        persona="A Catan player that plays based on a trained reinforcement learning model"
    )
    rl_player.rl_model_for_setup_phase.init_from_existing(original_player.model_for_setup_phase.get_model())
    rl_player.rl_model_for_trade_response.init_from_existing(original_player.model_for_responding_to_trade_requests.get_model())
    rl_model_gameplay = RLDecisionModel(structure=default_structure)
    # --- Load existing model if resuming ---
    rl_model_gameplay.model = tf.keras.models.load_model(
            "model.keras",
            custom_objects={"ppo_loss": ppo_loss},
            safe_mode=False
        )
    rl_player.rl_model_for_gameplay_phase = rl_model_gameplay

    # === evaluation ===
    tr = Tournament(no_games_in_tournament=24)
    tr.verbose = False
    players = default_players.copy()
    players[0] = rl_player
    rl_player.rl_model_for_gameplay_phase.explore = False
    overall_tournament_points, overall_victory_points, overall_rounds = tr.tournament(players)
    if args.verbose:
        tr.print_tournament_results(overall_tournament_points, overall_victory_points, overall_rounds, players)
    df_tournament_log = tr.log_tournament_results_in_dataframe(args.identifier,overall_tournament_points, overall_victory_points, overall_rounds, players, df_tournament_log)
    save_path = f"rl_tournament_log_{args.identifier}.csv"
    df_tournament_log.to_csv(save_path, index=False)
    s3.upload_file(save_path, bucket_name, f"results/{save_path}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="model.keras", help="Model name in /models/ on s3")
    parser.add_argument("--verbose", action="store_true", help="Print detailed tournament results")
    parser.add_argument("--identifier", type=int, default=2, help="Identifier for this run")



    args = parser.parse_args()
    main(args)