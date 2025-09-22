
import sys
import os
import numpy as np
sys.path.append("./src")
from Py_Catan_AI.default_structure import default_players
from Py_Catan_AI.game_log import victory_points_from_game_log, rounds_from_game_log
from Py_Catan_AI.py_catan_game import PyCatanGame
from Py_Catan_AI.model_based_catan_player import ModelBasedCatanPlayer
from Py_Catan_AI.rl_model_based_catan_player import RLModelBasedCatanPlayer
from Py_Catan_AI.rl_game_log import RLReplayBuffer
from Py_Catan_AI.py_catan_tournament import Tournament
from Py_Catan_AI.rl_tournament import RLTournament, debug_dataset
from Py_Catan_AI.rl_decision_model import RLDecisionModel
from Py_Catan_AI.default_structure import default_structure
from Py_Catan_AI.ppo_trainer import PPOTrainer
import tensorflow as tf


def policy_entropy_loss(y_true, y_pred):
    """
    Custom loss = categorical crossentropy (policy gradient) - beta * entropy
    - y_true: one-hot target (action taken)
    - y_pred: predicted probabilities
    """
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-9), axis=-1)
    return ce - 0.01 * entropy   # beta = 0.01 is a good starting point



no_rounds_of_training = 1  # adjust for longer runs
gamma = 0.99
number_of_games_in_tournament_training = 1
number_of_games_in_tournament_testing = 1
n_jobs = 1

"""
no_rounds_of_training = 10  # adjust for longer runs
gamma = 0.99
number_of_games_in_tournament_training = 48
number_of_games_in_tournament_testing = 24

Overall tournament results:
Player                  Avg Points      Std Points      Avg Victory Pts Std Victory Pts Avg Rounds      Std Rounds
RL Model Player         2.04            0.60            4.21            0.34            17.67           2.02
Hall 9000               4.62            0.72            5.88            0.36            17.67           2.02
Miss Minutes            5.06            0.79            6.29            0.35            17.67           2.02
C-3PO                   5.27            0.71            5.83            0.36            17.67           2.02
"""

original_player = ModelBasedCatanPlayer(name="original", persona="Original")
rl_player = RLModelBasedCatanPlayer(name="RL Model Player", persona="A Catan player that plays based on a trained reinforcement learning model")
rl_player.rl_model_for_setup_phase.init_from_existing(original_player.model_for_setup_phase.get_model())
rl_player.rl_model_for_trade_response.init_from_existing(original_player.model_for_responding_to_trade_requests.get_model())

# Keep one persistent model
rl_model_gameplay = RLDecisionModel(structure=default_structure)
rl_model_gameplay.init_from_existing(original_player.model_for_gameplay_phase.get_model())

# recompile with custom learning rate
# recompile with custom policy loss
rl_model_gameplay.model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={
        "output": policy_entropy_loss,   # custom loss instead of plain CE
        "value_output": "mse",
    },
    loss_weights={
        "output": 1.0,
        "value_output": 0.5,
    }
)

# Attach to RL player
rl_player.rl_model_for_gameplay_phase = rl_model_gameplay


# Training loop
for round_idx in range(no_rounds_of_training):

    print(f"\n=== Training round {round_idx+1}/{no_rounds_of_training} ===")

    # --- Collect data from tournament ---
    players = default_players.copy()
    players[0] = rl_player
    rl_player.rl_model_for_gameplay_phase.explore = True
    t = RLTournament(no_games_in_tournament=number_of_games_in_tournament_training)
    rl_log = t.tournament_rl_training_data_generation_parallel(players, gamma=0.99, n_jobs=n_jobs)
    dataset = t.to_training_dataset(rl_log, structure=default_structure)

    debug_dataset(dataset, name="gameplay_phase", model=rl_model_gameplay)
    print("After to_training_dataset:", np.mean(dataset["adv"]), np.std(dataset["adv"]))
    # # --- Normalize advantages ---
    # adv = np.array(dataset["adv"])
    # adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
    
    # ✅ Directly take normalized advantages from dataset
    adv = dataset["adv"]
    policy_weights = adv.reshape(-1)
    value_weights = np.ones_like(dataset["y_value"]).reshape(-1)

    # --- Train model (reusing the same instance) ---
    # === PPO Training ===
    ppo = PPOTrainer(
        rl_model=rl_model_gameplay,
        clip_epsilon=0.2,      # typical PPO value (0.1–0.3)
        entropy_coef=0.01,     # encourage exploration
        value_coef=0.5,        # balance value loss
        learning_rate=3e-4     # good default, tune if needed
    )

    ppo.train(dataset, epochs=5, batch_size=256)

    # --- Save model snapshot ---
    save_path = f"rl_model_gameplay_round{round_idx+1}.keras"
    rl_model_gameplay.model.save(save_path)
    print(f"✅ Saved {save_path}")





# 4. Evaluation step
tr = Tournament(no_games_in_tournament=number_of_games_in_tournament_testing)
tr.verbose = False
players = default_players.copy()
players[0] = rl_player
rl_player.rl_model_for_gameplay_phase.explore = False
overall_tournament_points, overall_victory_points, overall_rounds = tr.tournament(players)
tr.print_tournament_results(overall_tournament_points, overall_victory_points, overall_rounds, players)