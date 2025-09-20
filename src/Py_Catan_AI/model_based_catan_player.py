

import numpy as np
from Py_Catan_AI.catan_player import CatanPlayer
from Py_Catan_AI.decision_model import TrainedDecisionModelGamePlay, TrainedDecisionModelTradeResponse, TrainedDecisionModelGameSetup
from Py_Catan_AI.default_structure import default_structure

class ModelBasedCatanPlayer(CatanPlayer):
    def __init__(   self, 
                    name: str = 'Value Based Catan Player',
                    persona: str = "A Catan player that plays based on decisions by a neural network model",
                    model_for_setup_phase = None,
                    model_for_gameplay_phase = None,
                    model_for_responding_to_trade_requests = None
            ):
        if model_for_setup_phase is None:
            self.model_for_setup_phase = TrainedDecisionModelGameSetup()
        if model_for_gameplay_phase is None:
            self.model_for_gameplay_phase = TrainedDecisionModelGamePlay()
        if model_for_responding_to_trade_requests is None:
            self.model_for_responding_to_trade_requests = TrainedDecisionModelTradeResponse()
        super().__init__(structure = default_structure, name=name, persona=persona)

    def copy(self):
        return ModelBasedCatanPlayer(name=self.name, persona=self.persona)
    
    def decide_best_action(self,vector, mask):
        if mask[0] == 0:
            # this is the setup phase, so use different model
            active_model = self.model_for_setup_phase
        else:
            active_model = self.model_for_gameplay_phase
        best_action_index = active_model.create_prediction_from_vector_and_mask(vector=vector, mask=mask)

        return best_action_index
    
    def respond_positive_to_other_players_trading_request(self, vector, mask):
        '''        
        Return 'True' if this player responds positively to a trading request from another player.
        The vector has to be such that self is the first player, so rotate if needed.
        The mask should allow only the trade that is requested and the pass action (index 0)
        '''
        active_model = self.model_for_responding_to_trade_requests
        if sum(mask) == 1:
            return False
        else:
            options = np.where(mask==1)[0]
            proposed_action_index = options[options != 0][0] # get the trade option, not the pass option
            # decide based on value calculation if this trade is accepted
            best_action_index = active_model.create_prediction_from_vector_and_mask(vector = vector, mask = mask)
            return best_action_index == proposed_action_index