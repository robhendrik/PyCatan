# py_catan_rl_tournament.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Py_Catan_AI.py_catan_tournament import Tournament
from Py_Catan_AI.py_catan_game import PyCatanGame
from Py_Catan_AI.game_log import victory_points_from_game_log, rounds_from_game_log
from Py_Catan_AI.rl_game_log import RLReplayBuffer
import multiprocessing as mp


class RLTournamentParallel(Tournament):
    """
    Tournament class specialized for RL training data generation.
    Inherits basic tournament logic but adds helpers for RL logs.
    """
    def tournament_rl_training_data_generation_parallel(self, players, gamma=0.99, start_game_number=0, stop_game_number=24):
        """
        Play a tournament and collect RL training data with player order in [start_game_number: stop_game_number].
        The end-index is exclusive.

        Args:
            players: list of player instances (some may have rl_log attribute)
            gamma: discount factor for rewards (default 0.99)
            start_game_number: starting index of games to play (for parallelization)
            stop_game_number: stopping index of games to play (for parallelization)
        
        Returns:
            list of DataFrames with RL logs from all players and games played in this range.
        """
        players = [p.copy() for p in players]  # work on copies to avoid side effects

        all_rl_logs = []  # collect RL logs across games

        for game_number in range(start_game_number, stop_game_number):
            # === change order for every game to avoid same player always goes first
            players_for_this_game = self._order_elements(game_number, [p.copy() for p in players],reverse = False)

            # === Reset RL logs for players that have them
            for p in players_for_this_game:
                if hasattr(p, "rl_log"):
                    p.rl_log = RLReplayBuffer()  # reset RL log for new game

            # === play the game
            this_game = PyCatanGame(max_rounds=self.max_rounds_per_game, victory_points_to_win=self.victory_points_to_win)
            _ = this_game.play_catan_game(players = players_for_this_game)

            # === Collect RL logs if available
            for p in players_for_this_game:
                if hasattr(p, "rl_log"):
                    p.rl_log.finalize_rewards(gamma=gamma)
                    df = p.rl_log.to_dataframe()
                    df["game_number"] = game_number
                    df["player_name"] = p.name
                    all_rl_logs.append(df)

        return all_rl_logs
    



    

    

    
    def tournament_with_logs(self, players, gamma=0.99):
        """
        Play full tournament, collect and return merged RL logs.
        """
        tp, vp, rounds, rl_log = self.tournament_rl_training_data_generation(players, gamma=gamma)
        return tp, vp, rounds, rl_log

def split_logs_by_phase_parallel(rl_log: pd.DataFrame):
    """
    Split RL log DataFrame into separate logs per phase.
    Returns dict with keys: 'setup', 'gameplay', 'trade'.
    """
    logs = {}
    if rl_log is not None and not rl_log.empty:
        logs["setup"] = rl_log[rl_log["phase"] == "setup"].reset_index(drop=True)
        logs["gameplay"] = rl_log[rl_log["phase"] == "gameplay"].reset_index(drop=True)
        logs["trade"] = rl_log[rl_log["phase"] == "trade"].reset_index(drop=True)
    else:
        logs = {"setup": pd.DataFrame(), "gameplay": pd.DataFrame(), "trade": pd.DataFrame()}
    return logs

def save_training_log_parallel(rl_log: pd.DataFrame, filename="rl_log.pkl"):
    """
    Save RL training log to disk.
    """
    if rl_log is not None and not rl_log.empty:
        rl_log.to_pickle(filename)
        print(f"✅ RL training log saved to {filename} ({len(rl_log)} entries).")
    else:
        print("⚠️ Empty log, nothing saved.")

def load_training_log_parallel(filename="rl_log.pkl") -> pd.DataFrame:
    """
    Load RL training log from disk.
    """
    try:
        rl_log = pd.read_pickle(filename)
        print(f"✅ RL training log loaded from {filename} ({len(rl_log)} entries).")
        return rl_log
    except FileNotFoundError:
        print(f"⚠️ File {filename} not found.")
        return pd.DataFrame()

def to_training_dataset_parallel(rl_log_phase, structure, normalize_adv=True):
    """
    Convert a phase-specific RL log DataFrame into arrays for Keras training.
    
    Args:
        rl_log_phase: DataFrame with transitions for one phase (setup, gameplay, or trade).
        structure: game.structure object (to extract vector indices).
        normalize_adv: whether to normalize advantages (default True).
    
    Returns:
        A dict with numpy arrays:
            - x_inputs: [x1, x2, x3, mask] for the model
            - y_policy: one-hot action targets
            - y_value: return targets
            - adv: advantages (for custom policy loss)
    """
    if rl_log_phase is None:
        print("⚠️ Empty log, nothing to convert.")
        return None

    # Convert columns back to arrays
    states = np.stack(rl_log_phase["state"].values)
    masks = np.stack(rl_log_phase["mask"].values)
    actions = rl_log_phase["action"].values
    returns = np.array([float(r) for r in rl_log_phase["return"].values], dtype=np.float32).reshape(-1, 1)
    advantages = rl_log_phase["advantage"].values

    returns = rl_log_phase["return"].values.astype(np.float32)
    advantages = rl_log_phase["advantage"].values.astype(np.float32)

    # Verify advantage consistency with return - state_value
    if "state_value" in rl_log_phase.columns:
        recomputed_adv = returns - rl_log_phase["state_value"].values.astype(np.float32)
        diff = np.abs(recomputed_adv - advantages)
        if np.any(diff > 1e-4):
            print(f"⚠️ Advantage mismatch detected in {diff.sum()} entries "
                f"(max diff={diff.max():.5f})")


    # Prepare network inputs from states
    x1 = states[:, structure.vector_indices['nodes']]
    x2 = states[:, structure.vector_indices['edges']]
    x3 = states[:, structure.vector_indices['hands']].astype(np.float32)

    # One-hot encode actions
    num_actions = masks.shape[1]
    y_policy = np.zeros((len(actions), num_actions), dtype=np.float32)
    y_policy[np.arange(len(actions)), actions] = 1.0

    # Normalize advantages if requested
    if normalize_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset = {
        "x_inputs": [x1, x2, x3, masks],
        "y_policy": y_policy,
        "y_value": returns.astype(np.float32),
        "adv": advantages.astype(np.float32)
    }
    return dataset


def debug_dataset_parallel(dataset, name="dataset", model=None, max_points=5000):
    """
    Run sanity checks on the dataset before training PPO.
    Optionally run a forward pass with the given model.
    """
    print(f"\n🔍 Debugging {name} ...")

    # --- Advantage stats
    adv = dataset["adv"]
    print(" Advantage stats:")
    print("   mean:", np.mean(adv), " std:", np.std(adv),
          " min:", np.min(adv), " max:", np.max(adv))

    # Plot advantage distribution
    plt.hist(adv, bins=50, alpha=0.7, color="blue")
    plt.title(f"Advantage distribution ({name})")
    plt.xlabel("Advantage")
    plt.ylabel("Count")
    plt.show()

    # --- Policy one-hot encoding check
    y_policy = dataset["y_policy"]
    row_sums = y_policy.sum(axis=1)
    unique_sums = np.unique(row_sums)
    print(" Policy targets:")
    print("   shape:", y_policy.shape,
          " unique row sums (should all be 1):", unique_sums[:10])

    # --- Value targets
    y_value = dataset["y_value"]
    print(" Value targets:")
    print("   mean:", np.mean(y_value), " std:", np.std(y_value),
          " min:", np.min(y_value), " max:", np.max(y_value))

    # Plot returns distribution
    plt.hist(y_value, bins=50, alpha=0.7, color="green")
    plt.title(f"Returns/Value targets distribution ({name})")
    plt.xlabel("Return")
    plt.ylabel("Count")
    plt.show()

    # --- Forward pass sanity check
    if model is not None:
        try:
            logits, values = model.predict_logits_and_value(dataset["x_inputs"], verbose=0)
            probs = tf.nn.softmax(logits).numpy()
            print(" Policy probs:")
            print("   shape:", probs.shape,
                  " row sums (should be 1):", np.round(probs.sum(axis=1)[:10], 3))
            print(" Value preds:")
            print("   mean:", np.mean(values), " std:", np.std(values))
        except Exception as e:
            print("⚠️ Forward pass failed:", e)

    print(f"✅ Finished debugging {name}\n")
