import numpy as np
from Py_Catan_AI.catan_player import CatanPlayer
from Py_Catan_AI.rl_decision_model import RLDecisionModel
from Py_Catan_AI.default_structure import default_structure
from Py_Catan_AI.rl_game_log import RLReplayBuffer

class RLModelBasedCatanPlayer(CatanPlayer):
    def __init__(self, 
                 name: str = 'RL Catan Player',
                 persona: str = "A Catan player that plays based on RL decision models",
                 rl_model_for_setup_phase: RLDecisionModel = None,
                 rl_model_for_gameplay_phase: RLDecisionModel = None,
                 rl_model_for_trade_response: RLDecisionModel = None,
                 explore: bool = True):
        """
        explore=True means sample from action distribution (for training / self-play).
        explore=False means pick greedy best action (for evaluation).
        """
        if rl_model_for_setup_phase is None:
            self.rl_model_for_setup_phase = RLDecisionModel(default_structure)
        else:
            self.rl_model_for_setup_phase = rl_model_for_setup_phase

        if rl_model_for_gameplay_phase is None:
            self.rl_model_for_gameplay_phase = RLDecisionModel(default_structure)
        else:
            self.rl_model_for_gameplay_phase = rl_model_for_gameplay_phase

        if rl_model_for_trade_response is None:
            self.rl_model_for_trade_response = RLDecisionModel(default_structure)
        else:
            self.rl_model_for_trade_response = rl_model_for_trade_response

        self.explore = explore
        super().__init__(structure=default_structure, name=name, persona=persona)
        self.rl_log = RLReplayBuffer()
        # bool to force behavior or regular model based player
        self.mimic = False

    def copy(self):
        return RLModelBasedCatanPlayer(
            name=self.name, 
            persona=self.persona,
            rl_model_for_setup_phase=self.rl_model_for_setup_phase,
            rl_model_for_gameplay_phase=self.rl_model_for_gameplay_phase,
            rl_model_for_trade_response=self.rl_model_for_trade_response,
            explore=self.explore
        )

    def decide_best_action(self, vector, mask):
        """
        Use the RLDecisionModel to pick an action.
        - vector: board state vector (1D)
        - mask: valid actions (1D binary mask)
        """
        if self.mimic:
            return self.decide_best_action_mimic(vector,mask)
        
        if mask[0] == 0:
            active_model = self.rl_model_for_setup_phase
            phase = 'setup'
        else:
            active_model = self.rl_model_for_gameplay_phase
            phase = 'gameplay'

        action, probs, value = active_model.get_action(
            vector_row=vector, mask_row=mask, explore=self.explore
        )
        # Log decision
        self.rl_log.add_decision(
        state_vec=vector, mask=mask, action=action,
        probs=probs, value=value, phase=phase, player_name=self.name
        )
        return action

    def respond_positive_to_other_players_trading_request(self, vector, mask):
        """
        Return True if this player accepts a trade.
        - vector: board state vector (1D)
        - mask: trade mask (1D binary mask, includes pass option at index 0)
        """
        if self.mimic:
            return self.respond_positive_to_other_players_trading_request_mimic(vector,mask)
        
        active_model = self.rl_model_for_trade_response
        if sum(mask) == 1:
            return False
        else:
            options = np.where(mask == 1)[0]
            proposed_action_index = options[options != 0][0]  # the trade, not the pass
            action, probs, value = active_model.get_action(
                vector_row=vector, mask_row=mask, explore=self.explore
            )
            # Log trade decision
            self.rl_log.add_decision(
            state_vec=vector, mask=mask, action=action,
            probs=probs, value=value, phase="trade", player_name=self.name
            )

            return action == proposed_action_index
        
    def update_rl_log_game_information(self, round, action_in_round, score):
        """
        Update the RL log with game information after each action.
        - round: current round number
        - action_in_round: action index within the round
        - score: current victory points of the player
        """
        self.rl_log.update_game_info(round=round, action_in_round=action_in_round, score=score)

    def decide_best_action_mimic(self, vector, mask):
        """
        Behave exactly like the original DecisionModel player:
        - Greedy argmax from the policy head (no exploration).
        - Ignores value output.
        """
        if mask[0] == 0:
            active_model = self.rl_model_for_setup_phase
            phase = 'setup'
        else:
            active_model = self.rl_model_for_gameplay_phase
            phase = 'gameplay'

        # Force greedy
        action, probs, value = active_model.get_action(vector, mask, explore=False)
        
        # Log trade decision
        self.rl_log.add_decision(
        state_vec=vector, mask=mask, action=action,
        probs=probs, value=value, phase=phase, player_name=self.name
        )
        return action
    
    def respond_positive_to_other_players_trading_request_mimic(self, vector, mask):
        """
        Behave exactly like the original DecisionModel-based player:
        - Use greedy argmax from the policy head (no exploration).
        - Accept trade if chosen action == proposed trade action.
        """
        active_model = self.rl_model_for_trade_response
        if sum(mask) == 1:
            return False
        else:
            options = np.where(mask == 1)[0]
            proposed_action_index = options[options != 0][0]
            action, probs, value = active_model.get_action(
                vector_row=vector, mask_row=mask, explore=False
            )

            # Log trade decision
            self.rl_log.add_decision(
            state_vec=vector, mask=mask, action=action,
            probs=probs, value=value, phase="trade", player_name=self.name
            )
            return action == proposed_action_index



