"""RL model-based Catan player.

This module provides ``RLModelBasedCatanPlayer``, a concrete player
implementation that delegates decision-making to one or more
``RLDecisionModel`` instances. Decisions and related metadata are
recorded in an ``RLReplayBuffer`` for later training or analysis.

The class supports separate models for the setup phase, gameplay and
trade-response decisions. It also exposes utility methods to update the
per-step log with game-level metadata.

Author:
    Rob Hendriks

Version:
    1.0.0
"""

import numpy as np
from Py_Catan_AI.player import CatanPlayer
from Py_Catan_AI.rl_decision_model import RLDecisionModel
from Py_Catan_AI.default_structure import default_structure
from Py_Catan_AI.rl_game_log import RLReplayBuffer
from importlib.resources import files
from keras.models import load_model

class RLModelBasedCatanPlayer(CatanPlayer):
    """Player implementation that uses RL decision models.

    This player wraps up to three ``RLDecisionModel`` instances:
    - ``rl_model_for_setup_phase``: used when the game is in setup/placement
      (typically when mask[0] == 0 in this codebase).
    - ``rl_model_for_gameplay_phase``: used during normal gameplay.
    - ``rl_model_for_trade_response``: used when responding to trade offers.

    The player records each decision in an internal ``RLReplayBuffer``
    instance (``self.rl_log``) so callers can later convert the logs
    into a training dataset.

    Args:
        name: Human-readable player name.
        persona: Brief persona/description string.
        rl_model_for_setup_phase: Optional RLDecisionModel for setup.
        rl_model_for_gameplay_phase: Optional RLDecisionModel for gameplay.
        rl_model_for_trade_response: Optional RLDecisionModel for trades.
    """
    DEFAULT_PATH_TO_MODELS = 'Py_Catan_AI.models'
    PATH_TO_DEFAULT_TRADE_MODEL = files(DEFAULT_PATH_TO_MODELS).joinpath('rl_decision_default_trade_model.keras')
    PATH_TO_DEFAULT_SETUP_MODEL = files(DEFAULT_PATH_TO_MODELS).joinpath('rl_decision_default_setup_model.keras')
    PATH_TO_DEFAULT_GAMEPLAY_MODEL = files(DEFAULT_PATH_TO_MODELS).joinpath('rl_decision_default_gameplay_model.keras')
    
    def __init__(
        self,
        name: str = "RL Catan Player",
        persona: str = "A Catan player that plays based on RL decision models",
        rl_model_for_setup_phase: RLDecisionModel = None,
        rl_model_for_gameplay_phase: RLDecisionModel = None,
        rl_model_for_trade_response: RLDecisionModel = None,
    ):
        if rl_model_for_setup_phase is None:
            self.rl_model_for_setup_phase = RLDecisionModel(default_structure)
            self.rl_model_for_setup_phase.model = load_model(self.PATH_TO_DEFAULT_SETUP_MODEL, safe_mode=False)
            self.rl_model_for_setup_phase.explore = False
        else:
            self.rl_model_for_setup_phase = rl_model_for_setup_phase

        if rl_model_for_gameplay_phase is None:
            self.rl_model_for_gameplay_phase = RLDecisionModel(default_structure)
            self.rl_model_for_gameplay_phase.model = load_model(self.PATH_TO_DEFAULT_GAMEPLAY_MODEL, safe_mode=False)
            self.rl_model_for_gameplay_phase.explore = False
        else:
            self.rl_model_for_gameplay_phase = rl_model_for_gameplay_phase

        if rl_model_for_trade_response is None:
            self.rl_model_for_trade_response = RLDecisionModel(default_structure)
            self.rl_model_for_trade_response.model = load_model(self.PATH_TO_DEFAULT_TRADE_MODEL, safe_mode=False)
            self.rl_model_for_trade_response.explore = False
        else:
            self.rl_model_for_trade_response = rl_model_for_trade_response

        super().__init__(structure=default_structure, name=name, persona=persona)
        self.rl_log = RLReplayBuffer()

    def copy(self):
        """Create a shallow copy of the player preserving the RL models.

        The returned player references the same model instances (no
        deep-copy). This is convenient when starting multiple games that
        should share a policy.

        Returns:
            RLModelBasedCatanPlayer: new player instance with same models.
        """
        return RLModelBasedCatanPlayer(
            name=self.name,
            persona=self.persona,
            rl_model_for_setup_phase=self.rl_model_for_setup_phase,
            rl_model_for_gameplay_phase=self.rl_model_for_gameplay_phase,
            rl_model_for_trade_response=self.rl_model_for_trade_response,
        )

    def decide_best_action(self, vector, mask):
        """Decide an action using the appropriate RL model and log it.

        The function chooses the model based on the provided ``mask`` and
        delegates to ``RLDecisionModel.get_action``. The chosen action,
        policy probabilities and value estimate are logged to ``self.rl_log``.

        Args:
            vector: 1D state vector for the active player (numpy array-like).
            mask: 1D binary mask (or array-like) indicating valid actions.

        Returns:
            int: The selected action index.
        """
        if mask[0] == 0:
            active_model = self.rl_model_for_setup_phase
            phase = "setup"
        else:
            active_model = self.rl_model_for_gameplay_phase
            phase = "gameplay"

        action, probs, value, best_action, old_action_prob = active_model.get_action(
            vector_row=vector.copy(), mask_row=mask.copy(), include_best_action=True, include_old_action_prob=True
        )
        # Log decision
        self.rl_log.add_decision(
            state_vec=vector.copy(),
            mask=mask.copy(),
            action=action,
            probs=probs,
            value=value,
            phase=phase,
            player_name=self.name,
            best_action=best_action,
            old_action_prob=old_action_prob,
        )
        return action

    def respond_positive_to_other_players_trading_request(self, vector, mask):
        """Decide whether to accept a trade request and log the decision.

        Uses ``rl_model_for_trade_response`` to score the proposal. If the
        mask contains only the pass option the function returns ``False``.

        Args:
            vector: 1D state vector (numpy array-like).
            mask: 1D binary mask with pass option at index 0 and trade
                  option(s) at other indices.

        Returns:
            bool: ``True`` when the model chooses the proposed trade,
            ``False`` otherwise.
        """
        active_model = self.rl_model_for_trade_response
        if sum(mask) == 1:
            return False
        else:
            options = np.where(mask == 1)[0]
            proposed_action_index = options[options != 0][0]  # the trade, not the pass
            action, probs, value, best_action, old_action_prob = active_model.get_action(
                vector_row=vector, mask_row=mask, include_best_action=True, include_old_action_prob=True
            )
            # Log trade decision
            self.rl_log.add_decision(
                state_vec=vector,
                mask=mask,
                action=action,
                probs=probs,
                value=value,
                phase="trade",
                player_name=self.name,
                best_action=best_action,
                old_action_prob=old_action_prob,
            )

            return action == proposed_action_index

    def update_rl_log_game_information(self, round, action_in_round, score):
        """Attach per-step game metadata to the last logged decision.

        This should be called by the game loop right after a player's
        action has been applied so that the corresponding RL log entry
        contains the correct round, per-round action index and score.

        Args:
            round: Integer round counter at the time of the action.
            action_in_round: Integer index of the action within the round.
            score: Current victory points (integer) of the player.
        """
        self.rl_log.update_game_info(round=round, action_in_round=action_in_round, score=score)



