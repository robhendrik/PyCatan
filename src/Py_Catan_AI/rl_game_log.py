"""RL replay buffer utilities for reinforcement learning training.

This module provides ``RLReplayBuffer``, a small per-player buffer used
to collect state, action, policy and value predictions produced by an
RL agent during gameplay. The buffer stores per-step dictionaries that
can later be converted into a pandas DataFrame, augmented with
computed returns/advantages, and used to train policy/value networks.

The implementation is intentionally lightweight and stores plain Python
dicts so they can be serialized or post-processed easily.

Author:
    Rob Hendriks

Version:
    1.0.0
"""

import pandas as pd
from Py_Catan_AI.default_structure import default_structure
import numpy as np
from Py_Catan_AI.value_utils import calculate_value_for_first_player


class RLReplayBuffer:
    """Collect per-step RL training data for a single player.

    Each logged entry is a dictionary that typically contains at least:
    - ``state``: environment state vector (numpy array-like)
    - ``mask``: action availability mask
    - ``action``: integer action index taken
    - ``policy_probs``: action-probabilities produced by the policy
    - ``state_value``: scalar value prediction from the critic

    The buffer exposes convenience methods to update game metadata,
    compute heuristic returns, and to convert the stored entries to a
    ``pandas.DataFrame`` for downstream training or analysis.

    Note: the buffer does not itself perform any heavy tensor ops so
    it is safe to use from game loops; numerical processing is deferred
    to callers that convert the buffer into a DataFrame.
    """
    def __init__(self):
        self.entries = []

    def explanation(self):
        """Return a short human-readable explanation of the buffer contents.

        Returns:
            str: Multi-line description of the typical keys stored in each
            entry and their meaning. This is a convenience method used in
            debugging and documentation generation.
        """
        text = "RLReplayBuffer entries and short descriptions:\n\n"
        text += "state: The environment state vector for the active player (numpy array-like).\n"
        text += "mask: Boolean or int mask indicating which actions are valid for the active player.\n"
        text += "action: Integer index of the action actually taken by the agent.\n"
        text += "policy_probs: 1D array of action probabilities produced by the policy (sums to ~1).\n"
        text += "state_value: Scalar value estimate from the critic for the current state.\n"
        text += "reward: Immediate reward observed at this step (filled after game or scoring).\n"
        text += "phase: String describing the game phase when the decision was made (e.g. 'initial_placement' or 'gameplay').\n"
        text += "player: Player name or identifier for the owner of this entry.\n"
        text += "position_in_game: Integer (0-3) indicating the player's seat/order in the current game.\n"
        text += "round: Integer round counter at the time of the decision.\n"
        text += "action_in_round: Integer index of the action within the current round.\n"
        text += "score: Current victory point score for the active player at this step.\n"
        text += "game_indicator: Identifier or index for the game within a tournament/session.\n"
        text += "tournament_indicator: Identifier or index for the tournament (if applicable).\n"
        text += "meta_indicator: Miscellaneous meta flag used for bootstrap/auxiliary training markers.\n"
        text += "delta_reward: Per-step reward computed as the change in score from previous step.\n"
        text += "return: Discounted future cumulative reward from this step (to be computed in post-processing).\n"
        text += "advantage: Advantage estimate (return - state_value) used for policy gradient updates.\n"
        text += "scaled_heuristic_return: Optional potential-based return estimate computed from a heuristic.\n"
        text += "best_action: Optional best-action index recorded for analysis (e.g. greedy choice).\n"
        text += "game_UID: UUID string uniquely identifying the game instance.\n"
        text += "value_for_training: Optional scalar/target value prepared for training the critic.\n"
        text += "policy_for_training: Optional array/target policy prepared for training the policy network.\n"
        text += "final_ranking: Final placement/ranking of the player at game end (1-4).\n"
        text += "final_victory_points: Final victory points the player achieved at game end.\n"
        text += "old_action_prob: Probability under an older policy (useful for off-policy diagnostics).\n"
        return text
    
    def add_decision(self, state_vec, mask, action, probs, value, phase="gameplay", player_name=None, best_action = None, old_action_prob = None):
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
            "position_in_game": None,  # to be filled later
            "round": None,   # to be filled later
            "action_in_round": None,  # to be filled later
            "score": None,    # to be filled later
            "game_indicator": None,   # to be filled later
            "tournament_indicator": None,   # to be filled later
            "meta_indicator": None,   # to be filled later
            "delta_reward": None,   # to be filled later
            "return": None,   # to be filled later
            "advantage": None,   # to be filled later
            "scaled_heuristic_return": None,   # to be filled later
            "best_action": int(best_action) if best_action is not None else None,
            "game_UID": None,  # to be filled later
            "value_for_training": None,  # to be filled later
            "policy_for_training": None,  # to be filled later
            "final_ranking": None,  # to be filled later
            "final_victory_points": None,  # to be filled later
            "old_action_prob": float(old_action_prob) if old_action_prob is not None else None
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

    def add_player_name(self, player_name: str):
        """
        Add player name to all entries.
        """
        for entry in self.entries:
            entry["player"] = player_name
        return
    
    def add_final_ranking_and_victory_points(self, final_ranking: int, victory_points: int):
        """
        Add final ranking and final victory points to all entries.
        """
        for entry in self.entries:
            entry["final_ranking"] = int(final_ranking)
            entry["final_victory_points"] = int(victory_points)
        return
    
    def add_position_in_game_order(self, position_in_game: int):
        """
        Add the player's position in the game order (0-3) to all entries.
        """
        for entry in self.entries:
            entry["position_in_game"] = int(position_in_game)
        return
    
    

    def to_future_return_from_heuristic_value_scaled_for_game(self, structure=default_structure):
        """
        Approximate future returns using a potential-based heuristic function and updates entry['scaled_heuristic_return'].
        
        Args:
            structure: game.structure object (to extract vector indices).
        
        """
        heuristics = np.array([calculate_value_for_first_player(structure, entry["state"]) for entry in self.entries], dtype=np.float32)
        # Loop per entry
        for entry, heuristic_in_round in zip(self.entries, heuristics):
            final_val = heuristics[-1]
            remaining = final_val - heuristic_in_round
            entry['scaled_heuristic_return'] = (remaining -0.75)*2
        return 

    def to_dataframe(self):
        """
        Convert buffer to a DataFrame for training.
        """
        return pd.DataFrame(self.entries)
    
    def add_game_UID(self):
        """
        Add a unique game identifier to each entry.
        """
        import uuid
        game_uid = str(uuid.uuid4())
        for entry in self.entries:
            entry["game_UID"] = game_uid

# def finalize_rewards(self, gamma=1.0):
    #     """
    #     Compute per-step rewards, discounted returns, and advantages for training.
    #     - gamma: discount factor (1.0 = no discount)

    #     Each entry will have:
    #     - delta_reward: immediate score change at that step
    #     - return: discounted future sum of rewards from that step
    #     - advantage: return - state_value (for policy gradient training)


    #     score → cumulative VP

    #     delta_reward → per-step reward

    #     return → discounted sum of future rewards

    #     state_value → model’s predicted value (already logged)

    #     advantage → return minus baseline
    #     """
    #     print("Warning: Finalizing rewards in RLReplayBuffer...")
    #     if not self.entries:
    #         return

    #     rounds = [entry["round"] for entry in self.entries]
    #     scores = [entry["score"] for entry in self.entries]

    #     # per-step reward = score change
    #     delta_rewards = [scores[i] - (scores[i-1] if i > 0 else 0) for i in range(len(scores))]

    #     returns = [0.0] * len(self.entries)
    #     G = 0.0
    #     future_round = rounds[-1]

    #     # walk backwards through entries
    #     for i in reversed(range(len(self.entries))):
    #         round_gap = future_round - rounds[i]
    #         G = delta_rewards[i] + (gamma ** round_gap) * G
    #         returns[i] = G
    #         future_round = rounds[i]

    #     # assign back as plain floats
    #     for entry, d_r, ret in zip(self.entries, delta_rewards, returns):
    #         entry["delta_reward"] = float(d_r)
    #         entry["return"] = float(ret)
    #         # advantage requires a baseline if available
    #         sv = float(entry.get("state_value", 0.0))  # ensure scalar
    #         entry["advantage"] = float(ret) - sv