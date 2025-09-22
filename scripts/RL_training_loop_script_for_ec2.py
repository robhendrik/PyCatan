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

s3 = boto3.client("s3")
bucket_name = "pycatanbucket"

def save_to_s3(local_path, bucket, s3_path):
    s3.upload_file(local_path, bucket, s3_path)
    # print(f"✅ Uploaded {local_path} to s3://{bucket}/{s3_path}")

# python scripts/RL_training_loop_script_for_ec2.py --rounds 2 --train_games 1 --test_games 1  
# python scripts/RL_training_loop_script_for_ec2.py --rounds 2 --train_games 2 --test_games 1
# python scripts/RL_training_loop_script_for_ec2.py --rounds 2 --train_games 12 --test_games 1
# python scripts/RL_training_loop_script_for_ec2.py --rounds 2 --train_games 8 --test_games 2 --parallel --n_jobs 2

#(env_pycatan) C:\Users\nly99857\OneDrive - Philips\SW Projects\PyCatan>python scripts/RL_training_loop_clean_script.py --rounds 5 --train_games 1 --test_games 1

@tf.keras.utils.register_keras_serializable()
def policy_entropy_loss(y_true, y_pred):
    """Categorical crossentropy - entropy bonus (β=0.01)."""
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-9), axis=-1)
    return ce - 0.01 * entropy


def main(args):
    # --- Setup players ---
    original_player = ModelBasedCatanPlayer(name="original", persona="Original")
    rl_player = RLModelBasedCatanPlayer(
        name="RL Model Player",
        persona="A Catan player that plays based on a trained reinforcement learning model"
    )
    rl_player.rl_model_for_setup_phase.init_from_existing(original_player.model_for_setup_phase.get_model())
    rl_player.rl_model_for_trade_response.init_from_existing(original_player.model_for_responding_to_trade_requests.get_model())

    # --- RL Gameplay model ---
    rl_model_gameplay = RLDecisionModel(structure=default_structure)

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        rl_model_gameplay.model = tf.keras.models.load_model(
            args.resume,
            custom_objects={"ppo_loss": ppo_loss},
            safe_mode=False
        )
    else:
        print("Starting fresh RL model from original weights")
        rl_model_gameplay.init_from_existing(original_player.model_for_gameplay_phase.get_model())
        rl_model_gameplay.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss={"output": policy_entropy_loss, "value_output": "mse"},
            loss_weights={"output": 1.0, "value_output": 0.1}
        )

    rl_player.rl_model_for_gameplay_phase = rl_model_gameplay

    df_tournament_log = None

    # === Training loop ===
    for round_idx in range(args.rounds):
        print(f"\n=== Training round {round_idx+1}/{args.rounds} ===")

        players = default_players.copy()
        players[0] = rl_player
        rl_player.rl_model_for_gameplay_phase.explore = True

        t = RLTournament(no_games_in_tournament=args.train_games)
        t.verbose = True
        if args.parallel:
            tp, vp, rounds, rl_log = t.tournament_rl_training_data_generation_parallel(players, gamma=args.gamma, n_jobs=args.n_jobs)
        else:
            list_of_logs = []
            for n in range((args.train_games//24) + 1):
                t = RLTournament(no_games_in_tournament=24)
                print(f"--- Sub-tournament {n+1}, games {n*24+1}-{(n+1)*24} ---")
                tp, vp, rounds, rl_log = t.tournament_rl_training_data_generation(players, gamma=args.gamma)
                list_of_logs.append(rl_log)
            rl_log = pd.concat(list_of_logs, ignore_index=True)
        dataset = t.to_training_dataset(rl_log, structure=default_structure)

        #
        #debug_dataset(dataset, name="gameplay_phase", model=rl_model_gameplay)
        print("Dataset prepared.")
        # === PPO training ===
        ppo = PPOTrainer(
            rl_model=rl_model_gameplay,
            clip_ratio=0.2,
            #entropy_coef=0.01,
            #value_coef=0.5,
            learning_rate=args.lr
        )
        ppo.train(dataset, epochs=10, batch_size=256)
        print("PPO training complete.")
        # Save checkpoint
        save_path = f"rl_model_gameplay_round{round_idx+1}.keras"
        rl_model_gameplay.model.save(save_path, include_optimizer=False)
        # print(f"✅ Saved {save_path}")

        # Save to S3
        save_to_s3(save_path, bucket_name, f"models/{save_path}")
        
        # === evaluation ===
        tr = Tournament(no_games_in_tournament=args.test_games)
        tr.verbose = False
        players = default_players.copy()
        players[0] = rl_player
        rl_player.rl_model_for_gameplay_phase.explore = False
        overall_tournament_points, overall_victory_points, overall_rounds = tr.tournament(players)
        tr.print_tournament_results(overall_tournament_points, overall_victory_points, overall_rounds, players)
        df_tournament_log = tr.log_tournament_results_in_dataframe(round_idx+1,overall_tournament_points, overall_victory_points, overall_rounds, players, df_tournament_log)
        save_path = f"rl_tournament_log{round_idx+1}.csv"
        df_tournament_log.to_csv(save_path, index=False)
        # print(f"✅ Saved {save_path}")
        save_to_s3(save_path, bucket_name, f"results/{save_path}")
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10, help="Number of training rounds")
    parser.add_argument("--train_games", type=int, default=96, help="Games per training tournament")
    parser.add_argument("--test_games", type=int, default=24, help="Games per test tournament")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n_jobs", type=int, default=1, help="Processes for parallel mode")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel tournament execution")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to RL model checkpoint (.keras) to resume from")

    args = parser.parse_args()
    main(args)

