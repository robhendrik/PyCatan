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
from Py_Catan_AI.rl_tournament_parallel import RLTournamentParallel, to_training_dataset_parallel
from Py_Catan_AI.ppo_trainer import PPOTrainer
from Py_Catan_AI.ppo_trainer import ppo_loss
import multiprocessing as mp

import tensorflow as tf
import keras
import boto3

# python scripts/RL_training_loop_script_for_ec2_new_parallel.py --parallel

s3 = boto3.client("s3")
bucket_name = "pycatanbucket"

def find_optimal_assignment(train_games, n_jobs):
    if train_games <= 8:
        return [[(0,train_games)]]
    total_games = []
    for iterations in range(1, 1 + max(2, train_games // 8)):
        for nc in [1,2,4,8,16]:
            if nc > n_jobs:
                continue
            for games in [8,12,24]:
                if nc*games*iterations >= train_games:
                    total_games.append((nc*games*iterations,nc,games,iterations))

    total_games = sorted(total_games, key=lambda x: x[2]*x[3])
    solution = total_games[0]
    start, stop = 0, solution[2]
    assignments = [[] for _ in range(solution[1])]
    core_idx = 0
    while stop <= solution[0]:
        assignments[core_idx].append((start%24, start%24 + (stop-start)))
        start = stop
        stop += solution[2]
        core_idx = (core_idx + 1) % solution[1]
    return assignments

def worker_job(worker_assignment, worker_idx, working_model, gamma, list_of_logs):
    for start, stop in worker_assignment:
        print(f"Worker {worker_idx} assigned games {start+1}-{stop}")

        # === create rl player and load models ===
        original_player = ModelBasedCatanPlayer(name="original", persona="Original")
        rl_player = RLModelBasedCatanPlayer(
            name="RL Model Player",
            persona="A Catan player that plays based on a trained reinforcement learning model"
        )
        rl_player.rl_model_for_setup_phase.init_from_existing(original_player.model_for_setup_phase.get_model())
        rl_player.rl_model_for_trade_response.init_from_existing(original_player.model_for_responding_to_trade_requests.get_model())
        working_model = tf.keras.models.load_model(
            working_model,
            safe_mode=False
        )
        rl_player.rl_model_for_gameplay_phase.init_from_existing(working_model)
        players = default_players.copy()
        players[0] = rl_player
        # === Create tournament ===
        t = RLTournamentParallel()
        t.verbose = False
        # === Run tournament and collect logs ===
        rl_log = t.tournament_rl_training_data_generation_parallel(players, gamma=gamma, start_game_number=start, stop_game_number=stop)
        list_of_logs.extend(rl_log)
        return rl_log

@tf.keras.utils.register_keras_serializable()
def policy_entropy_loss(y_true, y_pred):
    """Categorical crossentropy - entropy bonus (β=0.01)."""
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-9), axis=-1)
    return ce - 0.01 * entropy

def main(args):
    print('Available cores:', os.cpu_count())
    print(f'Request: number of games {args.train_games}, n_jobs: {args.n_jobs}')
    assignments = find_optimal_assignment(args.train_games, args.n_jobs)
    original_player = ModelBasedCatanPlayer(name="original", persona="Original")
    # download working model from s3
    s3.download_file(bucket_name, f"models/{args.working_model}", "model.keras")

    # run jobs
    list_of_logs = []
    # for worker_idx, worker_assignment in enumerate(assignments):
    #     print(f"=== Worker {worker_idx}, assignment {worker_assignment} ===")
    #     worker_job(worker_assignment, worker_idx, "model.keras", list_of_logs)
    with mp.Pool(processes=len(assignments)) as pool:
        rl_logs = pool.starmap(worker_job, [(worker_assignment, worker_idx, "model.keras", args.gamma, []) for worker_idx, worker_assignment in enumerate(assignments)])
    for rl_log in rl_logs:
        list_of_logs.extend(rl_log)
    # merge logs and create dataset
    rl_log = pd.concat(list_of_logs, ignore_index=True)
    dataset = to_training_dataset_parallel(rl_log, structure=default_structure)
    print(f"Created training dataset with {len(dataset)} entries from {len(rl_log)} log entries.")
    # execute training
    initial_model = RLDecisionModel(structure = default_structure)
    saved_model = tf.keras.models.load_model(
                args.working_model,
                custom_objects={"ppo_loss": ppo_loss},
                safe_mode=False
            )
    initial_model.init_from_existing(saved_model)
    ppo = PPOTrainer(
        rl_model=initial_model,
        clip_ratio=0.2,
        learning_rate=args.lr
    )
    ppo.train(dataset, epochs=10, batch_size=256)

    # === Save checkpoint ===
    initial_model.model.save(args.working_model, include_optimizer=False)
    s3.upload_file(args.working_model, bucket_name, f"models/{args.working_model}")
    print(f"✅ Uploaded model checkpoint to s3://{bucket_name}/models/{args.working_model}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--working_model", type=str, default="working_model.keras", help="Path to RL model checkpoint (.keras) to resume from")
    parser.add_argument("--train_games", type=int, default=12, help="Games per training tournament")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n_jobs", type=int, default=4, help="Processes for parallel mode")

    args = parser.parse_args()
    main(args)

