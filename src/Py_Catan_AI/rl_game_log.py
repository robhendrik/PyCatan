

# rl_replay_buffer.py
import pandas as pd

class RLReplayBuffer:
    """
    Collects state/action/prob/value logs for RL training.
    Runs in parallel to the normal GameLog.
    """
    def __init__(self):
        self.entries = []

    def add_decision(self, state_vec, mask, action, probs, value, phase="gameplay", player_name=None):
        """
        Log a decision made by the RL player.
        """
        self.entries.append({
            "state": state_vec.copy() if hasattr(state_vec, "copy") else state_vec,
            "mask": mask.copy() if hasattr(mask, "copy") else mask,
            "action": int(action),
            "policy_probs": probs.copy(),
            "state_value": float(value),
            "reward": None,   # to be filled at end of game
            "phase": phase,
            "player": player_name,
            "round": None,   # to be filled later
            "action_in_round": None,  # to be filled later
            "score": None    # to be filled later
        })

    def update_game_info(self, round, action_in_round, score):
        """
        Update the last logged decision with game info.
        Should be called after each action in the game loop.
        """
        if self.entries:
            self.entries[-1]["round"] = round
            self.entries[-1]["action_in_round"] = action_in_round
            self.entries[-1]["score"] = score

    def finalize_rewards(self, gamma=1.0):
        """
        Compute per-step rewards, discounted returns, and advantages for training.
        - gamma: discount factor (1.0 = no discount)

        Each entry will have:
        - delta_reward: immediate score change at that step
        - return: discounted future sum of rewards from that step
        - advantage: return - state_value (for policy gradient training)


        score → cumulative VP

        delta_reward → per-step reward

        return → discounted sum of future rewards

        state_value → model’s predicted value (already logged)

        advantage → return minus baseline
        """
        if not self.entries:
            return

        rounds = [entry["round"] for entry in self.entries]
        scores = [entry["score"] for entry in self.entries]

        # per-step reward = score change
        delta_rewards = [scores[i] - (scores[i-1] if i > 0 else 0) for i in range(len(scores))]

        returns = [0.0] * len(self.entries)
        G = 0.0
        future_round = rounds[-1]

        # walk backwards through entries
        for i in reversed(range(len(self.entries))):
            round_gap = future_round - rounds[i]
            G = delta_rewards[i] + (gamma ** round_gap) * G
            returns[i] = G
            future_round = rounds[i]

        # assign back as plain floats
        for entry, d_r, ret in zip(self.entries, delta_rewards, returns):
            entry["delta_reward"] = float(d_r)
            entry["return"] = float(ret)
            # advantage requires a baseline if available
            sv = float(entry.get("state_value", 0.0))  # ensure scalar
            entry["advantage"] = float(ret) - sv




    def to_dataframe(self):
        """
        Convert buffer to a DataFrame for training.
        """
        return pd.DataFrame(self.entries)
