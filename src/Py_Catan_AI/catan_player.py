import numpy as np

from Py_Catan_AI.value_utils import calculate_value_for_first_player, calculate_value_hand_first_player_to_optimize_for_building_something
from Py_Catan_AI.vector_utils import execute_action_on_vector_for_first_player
from Py_Catan_AI.personas import MarvinTheParanoidAndroid, HAL9000, MissMinutes, C3PO


class CatanPlayer():
    def __init__(self, structure, name: str = 'Catan Player', persona: str = "A Catan player"):
        self.name = name
        self.persona = persona
        self.structure = structure
        self.atol = 0.001
        pass

    def copy(self):
        return CatanPlayer(self.structure, name=self.name, persona=self.persona)

    def decide_best_action(self,vector, mask):
        build_indices = np.array(self.structure.mask_indices['streets']).astype(np.int64)
        build_indices = np.concatenate((build_indices, np.array(self.structure.mask_indices['villages']).astype(np.int64)))
        build_indices = np.concatenate((build_indices, np.array(self.structure.mask_indices['towns']).astype(np.int64)))
        options = build_indices[mask[build_indices]==1]
        if len(options) > 0:
            action_index = np.random.choice(options)
        else:
            options = np.where(mask==1)[0]
            values = []

            for option in options:
                new_vector = execute_action_on_vector_for_first_player(self.structure, vector, option)
                value = calculate_value_hand_first_player_to_optimize_for_building_something(self.structure, new_vector)
                values.append(value)
            best_index = np.argmax(values)

            action_index = options[best_index]
        return action_index
    
    def respond_positive_to_other_players_trading_request(self, vector, mask):
        '''        
        Return 'True' if this player responds positively to a trading request from another player.
        The vector has to be such that self is the first player, so rotate if needed.
        The mask should allow only the trade that is requested, or the pass action (index 0)
        '''
        if sum(mask) == 1:
            return False
        else:
            options = np.where(mask==1)[0]
            trade_option = options[options != 0][0] # get the trade option, not the pass option
            # decide based on value calculation if this trade is accepted
            current_value = calculate_value_hand_first_player_to_optimize_for_building_something(self.structure, vector)
            new_vector = execute_action_on_vector_for_first_player(self.structure, vector, trade_option)
            new_value = calculate_value_hand_first_player_to_optimize_for_building_something(self.structure, new_vector)
        return (new_value > current_value) and not np.all(np.isclose(new_value,current_value, atol=self.atol))
    
class ValueBasedCatanPlayer(CatanPlayer):
    def __init__(self, structure, name: str = 'Value Based Catan Player', persona: str = "A Catan player that plays based on value calculations"):
        super().__init__(structure, name=name, persona=persona)

    def copy(self):
        return ValueBasedCatanPlayer(self.structure, name=self.name, persona=self.persona)
    
    def decide_best_action(self,vector, mask):
        options = np.where(mask==1)[0]
        values = []

        for option in options:
            new_vector = execute_action_on_vector_for_first_player(self.structure, vector, option)
            value = calculate_value_for_first_player(self.structure, new_vector)
            values.append(value)
        best_index = np.argmax(values)

        action_index = options[best_index]
        return action_index
    
    def respond_positive_to_other_players_trading_request(self, vector, mask):
        '''        
        Return 'True' if this player responds positively to a trading request from another player.
        The vector has to be such that self is the first player, so rotate if needed.
        The mask should allow only the trade that is requested and the pass action (index 0)
        '''
        if sum(mask) == 1:
            return False
        else:
            options = np.where(mask==1)[0]
            trade_option = options[options != 0][0] # get the trade option, not the pass option
            # decide based on value calculation if this trade is accepted
            current_value = calculate_value_for_first_player(self.structure, vector)
            new_vector = execute_action_on_vector_for_first_player(self.structure, vector, trade_option)
            new_value = calculate_value_for_first_player(self.structure, new_vector)
        return (new_value > current_value) and not np.all(np.isclose(new_value,current_value, atol=self.atol))
    

class RandomCatanPlayer(CatanPlayer):
    def __init__(self, structure, name: str = 'Random Catan Player', persona: str = "A Catan player that plays based on random moves"):
        super().__init__(structure, name=name, persona=persona)

    def copy(self):
        return RandomCatanPlayer(self.structure, name=self.name, persona=self.persona)
    
    def decide_best_action(self,vector, mask):
        options = np.where(mask==1)[0]
        action_index = np.random.choice(options)
        return action_index
    
    def respond_positive_to_other_players_trading_request(self, vector, mask):
        '''        
        Return 'True' if this player responds positively to a trading request from another player.
        The vector has to be such that self is the first player, so rotate if needed.
        The mask should allow only the trade that is requested and the pass action (index 0)
        '''
        if sum(mask) == 1:
            return False
        else:
            return np.random.choice([1,0])
        
class CompletelyPassiveCatanPlayer(CatanPlayer):
    def __init__(self, structure, name: str = 'Passive Catan Player', persona: str = "A Catan player that does nothing."):
        super().__init__(structure, name=name, persona=persona)

    def copy(self):
        return CompletelyPassiveCatanPlayer(self.structure, name=self.name, persona=self.persona)
    
    def decide_best_action(self,vector, mask):
        options = np.where(mask==1)[0]
        action_index = options[0] # always pass
        return action_index
    
    def respond_positive_to_other_players_trading_request(self, vector, mask):
        '''        
        Return 'True' if this player responds positively to a trading request from another player.
        The vector has to be such that self is the first player, so rotate if needed.
        The mask should allow only the trade that is requested and the pass action (index 0)
        '''
        return False
    
    
def default_names_and_personas():
    names = ['Marvin', 'Hall 9000', 'Miss Minutes', 'C-3PO']
    personas = [MarvinTheParanoidAndroid,
                HAL9000,
                MissMinutes,
                C3PO]
    return [(name, persona) for name, persona in zip(names, personas)]

def generate_default_players(structure):
    names_and_personas = default_names_and_personas()
    players = [ValueBasedCatanPlayer(structure, name=name, persona=persona) for (name, persona) in names_and_personas]
    return players

    