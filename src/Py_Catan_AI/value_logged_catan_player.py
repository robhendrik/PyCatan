"""Value-logged Catan player.

This module implements ``ValueLoggedCatanPlayer``, a value-based player
that additionally logs soft probability targets into an ``RLReplayBuffer``.
Instead of recording a one-hot action target, the player converts heuristic
action values into a smooth probability distribution. This is useful for
bootstrap training of policy-gradient algorithms where softer targets can
stabilize learning.

The module also provides a small helper ``_soft_probs_from_values`` that
turns a set of heuristic action values into a probability vector over the
full action space.

Author:
    Rob Hendriks

Version:
    1.0.0
"""

import numpy as np
from Py_Catan_AI.player import ValueBasedCatanPlayer
from Py_Catan_AI.rl_game_log import RLReplayBuffer
from Py_Catan_AI.vector_utils import execute_action_on_vector_for_first_player
from Py_Catan_AI.value_utils import calculate_value_for_first_player


def _soft_probs_from_values(values, legal_indices, action_space_len, tau=0.3, eps=1e-4):
    """Convert heuristic action values into a soft probability distribution.

    The function computes a temperature-scaled softmax over the provided
    heuristic values for the legal actions, applies epsilon smoothing and
    scatters the resulting probabilities into the full action space.

    Args:
        values: Sequence[float] of heuristic values for each legal action.
        legal_indices: Sequence[int] indices of legal actions in the
            full action space.
        action_space_len: int total number of possible actions.
        tau: float temperature (lower -> sharper distribution).
        eps: float small additive smoothing to avoid exact zeros.

    Returns:
        np.ndarray: probability vector of shape ``(action_space_len,)`` that
        sums to 1 and has non-zero mass only on ``legal_indices``.
    """
    vals = np.array(values, dtype=np.float32)

    # Numerical stability: subtract max
    z = (vals - np.max(vals)) / float(tau)
    e = np.exp(z)

    # Normalize over legal actions
    p_legal = e / np.sum(e)

    # Add epsilon smoothing, then renormalize
    p_legal = p_legal + eps
    p_legal = p_legal / p_legal.sum()

    # Scatter back into full action space
    probs = np.zeros(action_space_len, dtype=np.float32)
    probs[legal_indices] = p_legal
    return probs


class ValueLoggedCatanPlayer(ValueBasedCatanPlayer):
    """Value-based player that logs soft probability targets.

    This player uses the project's heuristic value function to pick an
    action (the highest-value legal action), but logs a soft probability
    distribution derived from the heuristic values to ``self.rl_log``.

    Parameters mirror ``ValueBasedCatanPlayer`` and an ``RLReplayBuffer``
    is created automatically on construction.
    """

    def __init__(self, structure, name: str = "ValueLogged Player",
                 persona: str = "A Catan player (value-based with RL logging)"):
        """Initialize the player and its RL log.

        Args:
            structure: Board structure object describing action indices and
                vector layout (usually ``default_structure``).
            name: Optional human-readable name.
            persona: Optional persona/description string.
        """
        super().__init__(structure, name=name, persona=persona)
        self.rl_log = RLReplayBuffer()
        # Temperature used when creating soft probability targets
        self.tau = 0.3

    def copy(self):
        """Return a shallow copy of the player with a fresh RL log.

        The returned object references the same value-based configuration
        but a new ``RLReplayBuffer`` is created when the player runs.
        """
        return ValueLoggedCatanPlayer(self.structure, name=self.name, persona=self.persona)

    def decide_best_action(self, vector, mask):
        """Pick a heuristic best action and log soft targets.

        Steps performed:
        1. Evaluate the heuristic value for each legal action.
        2. Select the highest-value action to play.
        3. Convert the set of values into a soft probability distribution
           (via ``_soft_probs_from_values``) and log the decision.

        Args:
            vector: 1D state vector for the active player.
            mask: 1D action mask (1 for legal actions).

        Returns:
            int: chosen action index.
        """
        options = np.where(mask == 1)[0]
        values = []

        # Evaluate each legal option
        for option in options:
            new_vector = execute_action_on_vector_for_first_player(self.structure, vector.copy(), option)
            value = calculate_value_for_first_player(self.structure, new_vector)
            values.append(value)

        # Choose the best action to actually play
        best_idx = int(np.argmax(values))
        action_index = int(options[best_idx])

        # before we had:
        # old version
        # # probs = np.zeros(len(mask), dtype=np.float32)
        # # probs[action_index] = 1.0
        
        # Soft probability distribution based on values
        probs = _soft_probs_from_values(values, options, len(mask), tau=self.tau, eps=1e-4)
        check_index = int(np.argmax(probs))
        
        if not check_index == action_index:
            print(f"Probabilities do not align with chosen action! The probabilities for the two indices are: {probs[action_index]} vs {probs[check_index]}. This is likely due to numerical issues.")
        if mask[0] == 0:
            phase = 'setup'
        else:
            phase = 'gameplay'

        # Log decision only if we have actions next to the pass action
        if not(sum(mask) == 1 and mask[0] == 1):
            self.rl_log.add_decision(
                state_vec=vector.copy(),
                mask=mask.copy(),
                action=action_index,
                probs=probs,
                value=float(values[best_idx]),  # log the heuristic value of the chosen action
                phase=phase,
                player_name=self.name,
                best_action=action_index
            )

        return action_index

    def respond_positive_to_other_players_trading_request(self, vector, mask):
        """Decide whether to accept a trade offer and log the decision.

        The method compares the heuristic value of the current state and the
        post-trade state. If the proposed trade increases the heuristic
        value it is accepted. A soft two-point distribution over ``pass``
        and ``trade`` is logged for training.

        Args:
            vector: 1D state vector for the active player.
            mask: 1D action mask including pass at index 0 and trade option(s).

        Returns:
            bool: True if the trade is accepted, False otherwise.
        """
        if sum(mask) == 1:
            action_index = 0
            current_value = calculate_value_for_first_player(self.structure, vector)
            probs = np.zeros(len(mask), dtype=np.float32)
            probs[0] = 1.0
            self.rl_log.add_decision(
                state_vec=vector.copy(),
                mask=mask.copy(),
                action=action_index,
                probs=probs,
                value=float(current_value),
                phase="trade",
                player_name=self.name,
                best_action=action_index
            )
            return False
        else:
            options = np.where(mask == 1)[0]
            trade_option = options[options != 0][0]

            current_value = calculate_value_for_first_player(self.structure, vector)
            new_vector = execute_action_on_vector_for_first_player(self.structure, vector, trade_option)
            new_value = calculate_value_for_first_player(self.structure, new_vector)

            # Accept if new value improves
            accept_trade = new_value > current_value #and not np.allclose(new_value, current_value, atol=self.atol)
            action_index = trade_option if accept_trade else 0

            # Create soft probs: if accepting, weight trade > pass; else reverse
            trade_vals = [current_value, new_value]
            trade_indices = [0, trade_option]
            probs = _soft_probs_from_values(trade_vals, trade_indices, len(mask), tau=0.5, eps=1e-4)
            best_action_from_probs = int(np.argmax(probs))
            if not best_action_from_probs == action_index:
                print(f"For {self.name} the probabilities do not align with chosen action! The probabilities for the two indices are: {probs[action_index]} vs {probs[best_action_from_probs]}. This is likely due to numerical issues.")
            # Log
            value_estimate = float(new_value if accept_trade else current_value)
            self.rl_log.add_decision(
                state_vec=vector.copy(),
                mask=mask.copy(),
                action=action_index,
                probs=probs,
                value=value_estimate,
                phase="trade",
                player_name=self.name,
                best_action=action_index
            )

            return accept_trade

    def update_rl_log_game_information(self, round, action_in_round, score):
        """Attach per-step metadata (round/action index/score) to the last log entry.

        This should be called from the game loop immediately after applying
        an action so the corresponding ``RLReplayBuffer`` entry is populated
        with the correct contextual information.

        Args:
            round: Integer round number.
            action_in_round: Integer index of the action within the round.
            score: Current victory points of the player.
        """
        self.rl_log.update_game_info(round=round, action_in_round=action_in_round, score=score)


