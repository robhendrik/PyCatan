import numpy as np

from Py_Catan_AI.board_structure import BoardStructure
from Py_Catan_AI.board_layout import BoardLayout
from Py_Catan_AI.vector_utils import reset_vector, mask_from_vector_for_building_village, execute_action_on_vector_for_first_player, rotate_vector_forward, rotate_vector_backward, vector_throw_dice, calculate_score_first_player
from Py_Catan_AI.vector_utils import mask_from_vector_for_building_street, mask_from_vector
from Py_Catan_AI.vector_utils import calculate_street_length_all_players, calculate_score_all_players
from Py_Catan_AI.default_structure import default_structure

# Game env does not know players names. It is fully based on the vector. 
# The index 0 in info['score'] and info['street_length'] always refers to the player with street 
# and village index 1, town index 5 and the hands in position vector_indices['hand_for_player'][0]
# The mask score and street_length are calculated after rotation, and before returning the vector,
# so the returned vector, scores etc are consistent (i.e., index 0 for player 1)
class PyCatanGameEnv():
    def __init__(self, structure = None, max_rounds=51, victory_points_to_win=10):
        if structure is not None:
            self.structure = structure
        else:
            self.structure = default_structure
        self.structure.max_rounds = max_rounds
        self.structure.winning_score = victory_points_to_win
        self.structure.plot_max_card_in_hand_per_type = 10
        self.player_A, self.player_B, self.player_C, self.player_D = 0,1,2,3      
        self.reset_game()
        return
    
    def reset_game(self):
        reward = 0
        self.vector, mask = reset_vector(self.structure)
        self.vector[self.structure.vector_indices['hand_for_player'][0]] = np.array(self.structure.real_estate_cost[0])
        mask = mask_from_vector_for_building_village(self.structure, self.vector)
        self.game_state = self._game_sequence()
        current_state = next(self.game_state)
        self.info = {'stage': current_state, 'rounds': 0, 'action in round': 0, 'dice result': 0,'terminated': False, 'truncated': False}
        self.info['street_length'] = [0,0,0,0] # for all players
        self.info['score'] = [0,0,0,0] # for all players
        self.info['mask'] = mask
        self.info['vector'] = self.vector
        return self.vector, mask, reward, self.info['terminated'], self.info['truncated'], self.info

    def step(self, action_index, trading_partner: int = None):
        reward = 0

        if action_index < 0:
            # pass to next action for same player
            self.info['action in round'] += 1
            self.info['dice result'] = 0
        elif self.info['stage']['phase'] == 'initial_placement':
            # execute the build action for village or street
            self.vector = execute_action_on_vector_for_first_player(self.structure, self.vector, action_index, trading_partner=trading_partner)
            # rotate vector if needed to close the current state
            if self.info['stage']['rotation'] == '+':
                self.vector = rotate_vector_forward(self.structure, self.vector)
            elif self.info['stage']['rotation'] == '-':
                self.vector = rotate_vector_backward(self.structure, self.vector)
            else:
                pass
            # move to next stage
            self.info['stage'] =  next(self.game_state)
            # create mask for next action
            if self.info['stage']['phase'] == 'initial_placement':
                if self.info['stage']['action_type'] == 'village':
                    self.vector[self.structure.vector_indices['hand_for_player'][0]] = np.array(self.structure.real_estate_cost[1])
                    mask = mask_from_vector_for_building_village(self.structure, self.vector)
                else:
                    self.vector[self.structure.vector_indices['hand_for_player'][0]] = np.array(self.structure.real_estate_cost[0])
                    mask = mask_from_vector_for_building_street(self.structure, self.vector, action_index)
                self.info['dice result'] = 0
            else:
                # this was the last step in intial placement, now create the mask for the first round in regular gameplay
                dice, self.vector = vector_throw_dice(self.structure, self.vector)
                self.info['dice result'] = dice
                mask = mask_from_vector(structure=self.structure, vector=self.vector)

        elif self.info['stage']['phase'] == 'game_play' and action_index > 0:
            # stay with same player in same stage and execute action
            self.vector = execute_action_on_vector_for_first_player(self.structure, self.vector, action_index, trading_partner=trading_partner)
            mask = mask_from_vector(structure=self.structure, vector=self.vector)   
            self.info['action in round'] += 1

            self.info['dice result'] = 0
        
        elif self.info['stage']['phase'] == 'game_play' and action_index == 0:
            self.info['stage'] =  next(self.game_state)
            self.vector = rotate_vector_forward(self.structure, self.vector)
            dice, self.vector = vector_throw_dice(self.structure, self.vector)
            self.info['dice result'] = dice
            mask = mask_from_vector(structure=self.structure, vector=self.vector)
            self.info['action in round'] = 0
            if self.info['stage']['active_player']== 0:
                self.info['rounds'] += 1
                if self.info['rounds'] > self.structure.max_rounds:
                    self.info['truncated'] = True

        self.info['street_length'] = calculate_street_length_all_players(self.structure, self.vector) # for all players
        self.info['score'] = calculate_score_all_players(self.structure, self.vector) # for all players
        if np.max(self.info['score']) >= self.structure.winning_score:
            self.info['terminated'] = True
        self.info['mask'] = mask
        self.info['vector'] = self.vector
        return self.vector, mask, reward, self.info['terminated'], self.info['truncated'], self.info

    def _game_sequence(self):
        game_order = { f'Player{p}_initial_placement_{at}0': 
                        {'active_player': p, 'phase': 'initial_placement', 'action_type': at, 'rotation':'+' if p < 3 and at == 'street' else '0'}
                        for p in [i for i in range(4)] for at in ['village', 'street'] }
        game_order.update({ f'Player{p}_initial_placement_{at}1':
                        {'active_player': p, 'phase': 'initial_placement', 'action_type': at, 'rotation':'-' if p > 0 and at == 'street' else '0'}
                        for p in [3-i for i in range(4)] for at in ['village', 'street']})
        game_order.update({ f'Player{p%4}_game_play': {'active_player': p%4, 'phase': 'game_play', 'rotation': '0'}
            for at in ['village', 'street'] for p in [i for i in range(5)] })
        next_keys = {list(game_order.keys())[i]: list(game_order.keys())[(i+1)] for i in range(len(game_order)-1)}
        next_keys.update({list(game_order.keys())[-1]: list(game_order.keys())[-4]})
        key = list(next_keys.keys())[0]
        while True:
            yield game_order[key]
            key = next_keys[key]


    




# 

