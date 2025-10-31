"""PyCatan Game Environment module for Catan gameplay simulation and AI training.

This module provides the core game environment for running Catan gameplay simulations
and training AI agents. It implements a complete game loop with state management,
action processing, and reward calculation using vectorized board representations.

The game environment operates on vector-based state representation and supports
both initial placement phases and regular gameplay phases. It handles player
rotation, dice rolling, action masking, and game termination conditions while
maintaining consistency between the game state and board vector representation.

The environment is designed to be compatible with reinforcement learning frameworks
and provides structured interfaces for AI agent training and evaluation.

Classes:
    PyCatanGameEnv: Main game environment class for Catan gameplay simulation.

Examples:
    Basic game environment setup:
        >>> env = PyCatanGameEnv()
        >>> vector, mask, reward, terminated, truncated, info = env.reset_game()
        >>> print(f"Game started with {len(vector)} state features")
        >>> print(f"Available actions: {np.sum(mask)}")

    Custom game configuration:
        >>> from Py_Catan_AI.board_structure import BoardStructure
        >>> from Py_Catan_AI.board_layout import BoardLayout
        >>> layout = BoardLayout(rings=3, winning_score=12)
        >>> structure = BoardStructure(layout)
        >>> env = PyCatanGameEnv(structure, max_rounds=100, victory_points_to_win=12)

    Running game steps:
        >>> env = PyCatanGameEnv()
        >>> env.reset_game()
        >>> # Select first valid action
        >>> valid_actions = np.where(env.info['mask'])[0]
        >>> action = valid_actions[0]
        >>> vector, mask, reward, terminated, truncated, info = env.step(action)
        >>> print(f"Round: {info['rounds']}, Score: {info['score']}")

Notes:
    The game environment uses a vector-based representation where:
    - Player indices in info correspond to rotated positions (index 0 = current player)
    - State vector contains board positions, player hands, and game metadata
    - Action masks indicate valid moves for the current player
    - Game progression follows standard Catan rules with initial placement and regular play
    
    Player indexing convention:
    - info['score'][0] and info['street_length'][0] refer to the current active player
    - Vector indices use fixed positions but data is rotated to maintain consistency
    - Player A,B,C,D are mapped to indices 0,1,2,3 respectively in the structure

Author:
    Rob Hendriks

Version:
    1.0.0
"""

import numpy as np

from Py_Catan_AI.vector_utils import reset_vector, mask_from_vector_for_building_village, execute_action_on_vector_for_first_player, rotate_vector_forward, rotate_vector_backward, vector_throw_dice, calculate_score_first_player
from Py_Catan_AI.vector_utils import mask_from_vector_for_building_street, mask_from_vector
from Py_Catan_AI.vector_utils import calculate_street_length_all_players, calculate_score_all_players
from Py_Catan_AI.default_structure import default_structure

class PyCatanGameEnv():
    """Catan game environment for AI training and gameplay simulation.
    
    This class implements a complete Catan game environment that manages game state
    using vectorized representations. It provides a standardized interface for AI
    agents to interact with the game, handling initial placement phases, regular
    gameplay, player rotation, and game termination conditions.
    
    The environment operates on a vector-based state representation that encodes
    the entire game state including board positions, player resources, scores,
    and available actions. Player perspectives are rotated so that the current
    active player is always at index 0 in returned information structures.
    
    Key Features:
        - Vector-based state representation for efficient AI processing
        - Automatic player rotation and turn management
        - Action masking for valid move generation
        - Support for both initial placement and regular game phases
        - Configurable game parameters (max rounds, victory conditions)
        - Comprehensive game state tracking and logging
    
    Game Flow:
        1. Initial placement phase: Each player places 2 villages and 2 roads
        2. Regular gameplay phase: Turn-based actions with dice rolling
        3. Game continues until victory condition or round limit reached
    
    State Vector Components:
        - Game metadata (turns remaining, player rankings)
        - Board occupation status (nodes, edges, tiles)
        - Player resource hands and scores
        - Action availability masks
    
    Player Indexing Convention:
        The environment uses a rotated indexing system where:
        - info['score'][0] always refers to the current active player
        - info['street_length'][0] always refers to the current active player
        - Player A,B,C,D are internally mapped to indices 0,1,2,3
        - Vector data is rotated to maintain consistency with current player perspective

    Game env does not know players names. It is fully based on the vector. The index 0 in 
    info['score'] and info['street_length'] always refers to the player with street and
    village index 1, town index 5 and the hands in position 
    vector_indices['hand_for_player'][0]. The mask score and street_length are calculated 
    after rotation, and before returning the vector, so the returned vector, scores etc 
    are consistent (i.e., index 0 for player 1).
    
    Attributes:
        structure (BoardStructure): Game board structure and configuration
        vector (np.ndarray): Current game state vector
        info (dict): Game information including scores, masks, and metadata
        game_state (generator): Internal game sequence generator
        player_A, player_B, player_C, player_D (int): Player index constants
    
    Examples:
        Basic environment setup:
            >>> env = PyCatanGameEnv()
            >>> vector, mask, reward, terminated, truncated, info = env.reset_game()
            >>> print(f"Initial state vector length: {len(vector)}")
            >>> print(f"Available actions: {np.sum(mask)}")
        
        Custom configuration:
            >>> from Py_Catan_AI.board_structure import BoardStructure
            >>> structure = BoardStructure()
            >>> env = PyCatanGameEnv(structure, max_rounds=100, victory_points_to_win=12)
            >>> env.reset_game()
        
        Game loop example:
            >>> env = PyCatanGameEnv()
            >>> env.reset_game()
            >>> while not env.info['terminated'] and not env.info['truncated']:
            ...     valid_actions = np.where(env.info['mask'])[0]
            ...     action = np.random.choice(valid_actions)
            ...     vector, mask, reward, terminated, truncated, info = env.step(action)
            ...     print(f"Round {info['rounds']}, Scores: {info['score']}")
    
    Notes:
        - The environment is stateful and maintains game progression internally
        - All returned information is from the perspective of the current active player
        - The vector representation enables efficient batch processing for AI training
        - Game rules follow standard Catan with initial placement and regular phases
        - Trading actions may require additional trading_partner parameter in step()
    """
    def __init__(self, structure = None, max_rounds=51, victory_points_to_win=10):
        """Initialize a new Catan game environment with specified configuration.
        
        Creates a new game environment instance with the given board structure
        and game parameters. The constructor sets up the board configuration,
        game rules, player mappings, and automatically initializes the first
        game by calling reset_game().
        
        The environment can be configured with custom board structures for
        different game variants or use the default standard Catan board.
        Game duration and victory conditions can be customized to support
        different training scenarios or game modes.
        
        Args:
            structure (BoardStructure, optional): Board structure defining the
                game layout, action spaces, and geometric relationships. If None,
                uses the default_structure for standard Catan configuration.
                Defaults to None.
            max_rounds (int, optional): Maximum number of rounds before the game
                is truncated. Each round consists of all 4 players taking their
                turns. Used to prevent infinite games during training.
                Defaults to 51.
            victory_points_to_win (int, optional): Victory points required to win
                the game. Standard Catan uses 10, but can be adjusted for shorter
                or longer games. Defaults to 10.
        
        Returns:
            None: Constructor initializes the instance in-place.
        
        Attributes Set:
            structure (BoardStructure): Game board and configuration
            player_A, player_B, player_C, player_D (int): Player index constants
            vector (np.ndarray): Current game state vector (set by reset_game)
            info (dict): Game information dictionary (set by reset_game)
            game_state (generator): Turn sequence generator (set by reset_game)
        
        Examples:
            Standard game environment:
                >>> env = PyCatanGameEnv()
                >>> print(f"Max rounds: {env.structure.max_rounds}")
                >>> print(f"Victory points needed: {env.structure.winning_score}")
            
            Custom configuration:
                >>> env = PyCatanGameEnv(max_rounds=100, victory_points_to_win=12)
                >>> print(f"Extended game: {env.structure.max_rounds} rounds")
            
            Custom board structure:
                >>> from Py_Catan_AI.board_structure import BoardStructure
                >>> from Py_Catan_AI.board_layout import BoardLayout
                >>> layout = BoardLayout(rings=3)
                >>> structure = BoardStructure(layout)
                >>> env = PyCatanGameEnv(structure=structure)
                >>> print(f"Board nodes: {env.structure.no_of_nodes}")
        
        Notes:
            - The constructor automatically calls reset_game() to initialize the first game
            - Structure parameters (max_rounds, winning_score) are set after structure assignment
            - Player indices are fixed: A=0, B=1, C=2, D=3 for internal consistency
            - The plot_max_card_in_hand_per_type is set to 10 for visualization purposes
            - If using a custom structure, ensure it's properly configured before passing
        
        Raises:
            No explicit exceptions, but reset_game() may print warnings for initialization issues.
        """
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
    
    def reset_game(self) -> tuple:
        """Reset the game environment to initial state for a new game.
        
        Initializes all game components to their starting state, including the
        board vector, player information, game sequence generator, and action
        masks. This method prepares the environment for the initial placement
        phase where the first player can place their first village.
        
        The reset process includes:
        - Resetting the board state vector to empty/initial configuration
        - Setting up player rankings and initial resource hands
        - Initializing the game sequence generator for turn management
        - Creating the initial action mask for village placement
        - Giving the first player the initial resources for village placement
        - Resetting all game tracking information (scores, rounds, etc.)
        
        Args:
            None: Uses instance attributes for configuration.
        
        Returns:
            tuple: A 6-tuple containing:
                - vector (np.ndarray): Initial game state vector
                - mask (np.ndarray): Boolean array of valid actions for first player
                - reward (float): Initial reward (always 0.0)
                - terminated (bool): Game termination status (always False)
                - truncated (bool): Game truncation status (always False) 
                - info (dict): Game information dictionary with keys:
                    - 'stage': Current game stage information
                    - 'rounds': Round counter (starts at 0)
                    - 'action in round': Action counter within round (starts at 0)
                    - 'dice result': Last dice roll result (starts at 0)
                    - 'terminated': Game completion flag
                    - 'truncated': Game truncation flag
                    - 'street_length': Road lengths for all players [0,0,0,0]
                    - 'score': Victory points for all players [0,0,0,0]
                    - 'mask': Action availability mask
                    - 'vector': Reference to game state vector
        
        Example:
            >>> env = PyCatanGameEnv()
            >>> vector, mask, reward, terminated, truncated, info = env.reset_game()
            >>> print(f"Game reset. Vector length: {len(vector)}")
            >>> print(f"Available actions: {np.sum(mask)}")
            >>> print(f"Current stage: {info['stage']['phase']}")
            >>> print(f"Active player: {info['stage']['active_player']}")
            
        Notes:
            - This method is called automatically during __init__()
            - The first player (index 0) starts with resources for village placement
            - Player rankings are verified to ensure proper initialization
            - The game begins in 'initial_placement' phase with 'village' action type
            - All players start with zero street length and victory points
        
        Raises:
            Warning: Prints warning if player rankings are not initialized correctly
        """
        reward = 0
        self.vector, mask = reset_vector(self.structure)
        if not np.array_equal(self.vector[self.structure.vector_indices['ranks']], np.array([0, 1, 2, 3])):
            print("Warning: ranks not initialized correctly in reset_vector")
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
        """Advance the game state by one action for the current active player.

        This method applies the provided action index (from the perspective of
        the player at index 0) to the internal vector representation, updates
        the game phase and rotation as required, rolls dice when a player ends
        their turn, and recalculates derived info such as scores and street
        lengths.

        Behavior summary:
        - action_index < 0: treated as a pass for the current player (advances
            the internal 'action in round' counter without modifying the vector).
        - During 'initial_placement' phase: executes the build action (village
            or street), handles rotation for closing placement steps, advances
            the internal stage generator, and prepares the next action mask. If
            the initial placement has finished, the first dice roll for regular
            gameplay is performed and an action mask for gameplay is produced.
        - During 'game_play' phase with action_index > 0: executes the chosen
            action for the current player, keeps the same active player and
            updates the in-round action counter.
        - During 'game_play' phase with action_index == 0: treats this as an
            end-of-turn for the active player: advances the stage generator,
            rotates the vector so the next player becomes index 0, rolls the
            dice for the arriving player, creates a new mask and resets the
            per-round action counter. When the active player cycles back to
            player 0 the global round counter is incremented and truncation is
            checked.

        Args:
                action_index (int): The action index (from the current player's
                        perspective) to execute. Negative values indicate a pass; 0 is
                        used to end the player's turn in gameplay; positive values
                        select a concrete action to execute.
                trading_partner (int | None): Optional index of a trading partner
                        used for trade-related actions. If not applicable, pass None.

        Returns:
                tuple: (vector, mask, reward, terminated, truncated, info)
                        - vector (np.ndarray): The updated game state vector after the
                            action and any rotations.
                        - mask (np.ndarray): Boolean action mask valid for the new
                            active player / current phase.
                        - reward (float): Reward for the step (currently always 0 in
                            the environment implementation).
                        - terminated (bool): True when a player has reached
                            `structure.winning_score` and the game has ended.
                        - truncated (bool): True when the game exceeded
                            `structure.max_rounds` and was truncated.
                        - info (dict): Full information dictionary containing keys such
                            as 'stage', 'rounds', 'action in round', 'dice result',
                            'street_length', 'score', 'mask', and 'vector'.

        Notes:
                - The method keeps the convention that index 0 in returned score
                    and street_length arrays refers to the current active player.
                - The function updates internal generator state returned by
                    `self._game_sequence()` and expects that generator to yield
                    stage dictionaries with keys 'phase', 'action_type', 'rotation'
                    and 'active_player'.
        """
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
        if not self.info['stage']['active_player']==self.vector[self.structure.vector_indices['ranks']][0]:
            print("Warning: active player in game state does not match rank 0 in vector")
        return self.vector, mask, reward, self.info['terminated'], self.info['truncated'], self.info

    def _game_sequence(self):
        """Yield the ordered sequence of game-stage dictionaries used by the env.

        This generator returns a sequence of stage dictionaries consumed by the
        environment to drive placement and gameplay. Each yielded dictionary
        describes the active player and what the upcoming action expects.

        Yielded dictionary keys
            - 'active_player' (int): player index whose turn/stage this is
            - 'phase' (str): either 'initial_placement' or 'game_play'
            - 'action_type' (str, optional): 'village' or 'street' for
              initial placement steps (omitted for game_play stages)
            - 'rotation' (str): one of '0', '+' or '-' indicating whether the
              environment should rotate the internal vector after the action to
              close/open the perspective for the next step.

        Sequence ordering
            1. First initial placement rounds: for players 0..3 each a 'village'
               then a 'street' placement (rotation '+' applied on some streets).
            2. Second initial placement rounds: players 3..0 reversed order
               with their second village/street placements (rotation '-' as
               appropriate for closing placement).
            3. Repeating 'game_play' stages that cycle players for regular turns.

        The generator loops indefinitely; callers consume it via next(self.game_state)
        to obtain the next stage dict. The `step()` method expects the yielded
        dicts to use the keys described above.

        Example:
            gs = self._game_sequence()
            first_stage = next(gs)
            # first_stage -> {'active_player': 0, 'phase': 'initial_placement', 'action_type': 'village', 'rotation': '0'}
        """
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


    






