"""Catan player implementations and helpers.

This module defines the player interface and several concrete player
implementations used in simulations and training. Players operate on the
environment's vectorized state and must provide decision callbacks used by the
game runner. The file also offers convenience factories for creating default
players and mapping names to persona classes.

Key classes
    - CatanPlayer: base class describing the player API (decide/respond)
    - ValueBasedCatanPlayer: decision-making based on heuristic value functions
    - RandomCatanPlayer: selects valid actions uniformly at random
    - CompletelyPassiveCatanPlayer: always passes / does nothing

Helper functions
    - default_names_and_personas(): returns standard (name, persona) tuples
    - generate_default_players(structure): builds a list of default players

Usage example
    >>> from Py_Catan_AI.catan_player import generate_default_players
    >>> players = generate_default_players(structure)

Notes
    - Player methods assume the provided `vector` argument is rotated so the
      current player is at index 0 whenever required by the decision logic.
    - Trade-response methods receive a `mask` that typically contains exactly
      one trade option and the pass action; implementations should inspect the
      mask accordingly.

Author: Rob Hendriks
Version: 1.0.0
"""

import numpy as np
from Py_Catan_AI.value_utils import calculate_value_for_first_player, calculate_value_hand_first_player_to_optimize_for_building_something
from Py_Catan_AI.vector_utils import execute_action_on_vector_for_first_player
from Py_Catan_AI.board_structure import BoardStructure
from Py_Catan_AI.personas import MarvinTheParanoidAndroid, HAL9000, MissMinutes, C3PO


class CatanPlayer():
    """Base player interface for PyCatan.

    This class documents the minimal API required by the game runner. It is
    lightweight and intended to be subclassed by concrete player policies used
    in simulations and training.

    Attributes
        structure (BoardStructure): Board layout and index helpers required by
            decision logic.
        name (str): Human-friendly player name used in logs.
        persona (str|object): Optional persona descriptor used for verbalization.
        atol (float): Numeric tolerance used in comparison operations (default 1e-3).

    Methods to implement in subclasses
        decide_best_action(vector: np.ndarray, mask: np.ndarray) -> int
            Return a valid action index (where mask[index] == 1) given the
            current (rotated) state vector and action mask.

        respond_positive_to_other_players_trading_request(vector: np.ndarray, mask: np.ndarray) -> bool
            Given a state vector (rotated so this player is at index 0) and a
            trade-response mask, return True to accept the trade or False to
            decline. Implementations should return False when only the pass
            action is available (i.e., sum(mask) == 1).

    Notes
        - The environment presents `vector` rotated so the active player is at
          index 0; player code should rely on this convention when inspecting
          resource/position fields.
        - Action semantics (indices) are defined by the `structure` object.

    Example
        >>> p = ValueBasedCatanPlayer(structure)
        >>> action = p.decide_best_action(env.vector, env.info['mask'])
    """
    def __init__(self, structure: BoardStructure, name: str = 'Catan Player', persona: str = "A Catan player"):
        """
        Initialize the Catan player.

        Args:
            structure (BoardStructure): The structure of the game board.
            name (str, optional): The name of the player. Defaults to 'Catan Player'.
            persona (str, optional): The persona of the player. Defaults to "A Catan player".
        """
        self.name = name
        self.persona = persona
        self.structure = structure
        self.atol = 0.001
        pass

    def copy(self):
        """
        Create a copy of the Catan player.

        Returns:
            CatanPlayer: A new CatanPlayer instance with the same attributes.
        """
        return CatanPlayer(self.structure, name=self.name, persona=self.persona)

    def decide_best_action(self, vector: np.ndarray, mask: np.ndarray) -> int:
        """
        Decide the best action for the player based on the current game state.

        Args:
            vector (np.ndarray): The current game state vector.
            mask (np.ndarray): The mask indicating valid actions.

        Returns:
            int: The index of the best action.
        """
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

    def respond_positive_to_other_players_trading_request(self, vector: np.ndarray, mask: np.ndarray) -> bool:
        """
        Return 'True' if this player responds positively to a trading request from another player.
        The vector has to be such that self is the first player, so rotate if needed.
        The mask should allow only the trade that is requested, or the pass action (index 0).

        Args:
            vector (np.ndarray): The current game state vector.
            mask (np.ndarray): The mask indicating valid actions.

        Returns:
            bool: True if the player accepts the trade, False otherwise.
        """
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
    """Value-based policy.

    Evaluates all legal actions by applying each to the current (rotated)
    state and scoring the resulting state with the project's heuristic
    value function. The action with the highest score is chosen. For trade
    responses, accepts a trade only when the heuristic value strictly
    increases.
    """
    def __init__(self, structure, name: str = 'Value Based Catan Player', persona: str = "A Catan player that plays based on value calculations"):
        """Initialize a ValueBasedCatanPlayer instance."""
        super().__init__(structure, name=name, persona=persona)

    def copy(self):
        """Create a copy of the ValueBasedCatanPlayer."""
        return ValueBasedCatanPlayer(self.structure, name=self.name, persona=self.persona)
    
    def decide_best_action(self,vector, mask):
        """Return the index of the best action to take. The best action is the one 
        that maximizes the value function after executing it."""
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
    """Random policy.

    Chooses uniformly at random from the legal actions. For trade responses
    the acceptance decision is also random (except when only the pass option
    is available, in which case it returns False).
    """
    def __init__(self, structure, name: str = 'Random Catan Player', persona: str = "A Catan player that plays based on random moves"):
        """ Initialize the RandomCatanPlayer. """
        super().__init__(structure, name=name, persona=persona)

    def copy(self):
        """ Create a copy of the RandomCatanPlayer. """
        return RandomCatanPlayer(self.structure, name=self.name, persona=self.persona)
    
    def decide_best_action(self,vector, mask):
        """ Return a random valid action index given the current (rotated) state vector and action mask."""
        options = np.where(mask==1)[0]
        action_index = np.random.choice(options)
        return action_index
    
    def respond_positive_to_other_players_trading_request(self, vector, mask):
        """
        Return 'True' if this player responds positively to a trading request from another player.
        The vector has to be such that self is the first player, so rotate if needed.
        The mask should allow only the trade that is requested and the pass action (index 0)
        """
        if sum(mask) == 1:
            return False
        else:
            return np.random.choice([1,0])
        
class CompletelyPassiveCatanPlayer(CatanPlayer):
    """Completely passive player used for testing and baselines.

    Always selects the first legal action (typically the pass action) and
    always declines any trade requests.
    """
    def __init__(self, structure, name: str = 'Passive Catan Player', persona: str = "A Catan player that does nothing."):
        """ Initialize the CompletelyPassiveCatanPlayer. """
        super().__init__(structure, name=name, persona=persona)

    def copy(self):
        """ Create a copy of the CompletelyPassiveCatanPlayer. """
        return CompletelyPassiveCatanPlayer(self.structure, name=self.name, persona=self.persona)
    
    def decide_best_action(self,vector, mask):
        """ Always return the first valid action index (typically the pass action)."""
        options = np.where(mask==1)[0]
        action_index = options[0] # always pass
        return action_index
    
    def respond_positive_to_other_players_trading_request(self, vector, mask):
        """ Always decline any trade requests. """
        return False
    
    
def default_names_and_personas():
    """ Return default names and personas for standard players. """
    names = ['Marvin', 'Hall 9000', 'Miss Minutes', 'C-3PO']
    personas = [MarvinTheParanoidAndroid,
                HAL9000,
                MissMinutes,
                C3PO]
    return [(name, persona) for name, persona in zip(names, personas)]

def generate_default_players(structure):
    """ Build a list of default players using standard names and personas. """
    names_and_personas = default_names_and_personas()
    players = [ValueBasedCatanPlayer(structure, name=name, persona=persona) for (name, persona) in names_and_personas]
    return players