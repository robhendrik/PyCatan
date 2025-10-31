"""Vector utilities and mask helpers for PyCatan.

This module contains helpers for creating and manipulating the environment
state "vector" and corresponding action masks used throughout the
project. Functions here build masks for legal actions, rotate the vector to
change player perspective, compute scores/lengths and apply action effects
(build street/village/town, throw dice, etc.).

Key functions
        - reset_vector(): create empty vector and mask for a new player/state
        - mask_from_vector(): construct legal-action mask for the first player
        - rotate_vector_forward/backward(): change player-perspective rotation
        - build_street/build_village/build_town(): apply build actions to a vector

Notes
        - Many functions expect the vector to be from the viewpoint of player 0
            (i.e., rotated so the active player is first).
        - Index/shape conventions are governed by `BoardStructure`.

Author: Rob Hendriks
Version: 1.0.0
"""

import numpy as np
from Py_Catan_AI.board_structure import BoardStructure





def reset_vector(structure) -> tuple[np.ndarray, np.ndarray]:
    """
    Reset the action and mask vectors for the given structure.

    Args:
        structure (BoardStructure): Board structure containing vector and mask indices.

    Returns:
        tuple[np.ndarray, np.ndarray]: empty vector and mask with only the first element set to 1
    """
    vector = np.zeros(structure.vector_indices['length'], dtype=np.int16)
    vector[structure.vector_indices['ranks']] = [i for i in range(len(structure.vector_indices['ranks']))]
    mask= np.zeros(structure.mask_indices['length'], dtype=np.int16)
    mask[0] = 1
    return vector, mask

def mask_from_vector(structure: BoardStructure,vector: np.ndarray) -> np.ndarray:
    """Construct the legal-action mask for the first player (player 0).

    Args:
        structure (BoardStructure): Board structure containing vector and mask indices.
        vector (np.ndarray): State vector rotated so player 0 is active.

    Returns:
        np.ndarray: Action mask for player 0 (dtype int16) with length equal to
            structure.mask_indices['length'].
    """
    vector_indices = structure.vector_indices
    mask_space_indices = structure.mask_indices
    hand = vector[vector_indices['hand_for_player'][0]]
    streets = vector[vector_indices['edges']] == 1
    free_edges_on_board = vector[vector_indices['edges']] == 0
    occupied_nodes = vector[vector_indices['nodes']] > 0
    free_nodes_on_board = np.logical_not(occupied_nodes @ structure.node_neighbour_matrix)
    villages = vector[vector_indices['nodes']] == 1
    towns = vector[vector_indices['nodes']] == 5

    mask= np.zeros(mask_space_indices['length'], dtype=np.int16)

    # pass is always a possible action
    pass_indices = mask_space_indices['pass']
    mask[pass_indices] = np.ones_like(pass_indices, dtype=np.int16)

    # Handle streets
    if np.all((hand - np.array(structure.real_estate_cost[0])) >= 0):
        if np.sum(streets) < structure.max_available_real_estate_per_type[0]:
            build_options_for_streets = np.logical_and(free_edges_on_board,streets @ structure.edge_edge_matrix )
            mask[mask_space_indices['streets']] = build_options_for_streets

    # Handle villages
    if np.all((hand - np.array(structure.real_estate_cost[1])) >= 0):
        if np.sum(villages) < structure.max_available_real_estate_per_type[1]:
            build_options_for_villages = np.logical_and(free_nodes_on_board,streets @ structure.edge_node_matrix )
            mask[mask_space_indices['villages']] = build_options_for_villages

    # Handle towns
    if np.all((hand - np.array(structure.real_estate_cost[2])) >= 0):
        if np.sum(towns) < structure.max_available_real_estate_per_type[2]:
            mask[mask_space_indices['towns']] = villages

    # Handle bank trades
    for resource, count in enumerate(hand):
        if count >= 4:
            mask[mask_space_indices['trades_bank']] = np.logical_or(mask[mask_space_indices['trades_bank']], structure.trade_options_array[:,0] == resource)

    # Handle player trades
    for resource, count in enumerate(hand):
        if count > 0:
            mask[mask_space_indices['trades_player']] = np.logical_or(mask[mask_space_indices['trades_player']], structure.trade_options_array[:,0] == resource)

    return mask

def mask_from_vector_for_responding_to_trade_request(structure: BoardStructure, vector: np.ndarray, trading_partner, proposed_trade_index) -> np.array:
    """Generate a response mask for a trading partner to a trade request.

    The returned mask always includes the pass action (index 0) and will
    include the accept-action index when the partner has the required resource.

    Args:
        structure (BoardStructure): Board metadata and index maps.
        vector (np.ndarray): Current state vector (not rotated further).
        trading_partner (int): Player index of the partner (0-3) in the vector.
        proposed_trade_index (int): Action index of the proposed trade (from
            the requestor's perspective); converted using
            ``structure.index_to_action`` inside the function.

    Returns:
        np.ndarray: Mask for the trading partner (dtype int16).
    """
    # pass is a possible action, so start with one 1
    mask= np.zeros(structure.mask_indices['length'], dtype=np.int16)
    mask[0] = 1

    proposed_action = structure.index_to_action(proposed_trade_index)
    if not proposed_action[0] == 'trade_player':
        raise Exception(f'Wrong action for trade request. Expected trade_player, got {proposed_action[0]}')
    proposed_trade = proposed_action[1]
    # check if the trading partner has the resources to accept the trade
    hand = vector[structure.vector_indices['hand_for_player'][trading_partner]]
    if hand[proposed_trade[1]] > 0:
        action_index = structure.action_to_index(('trade_player', (proposed_trade[1], proposed_trade[0]) ))
        mask[action_index] = 1
        return mask
    else:
        return mask

        


def mask_from_vector_for_building_street(structure: BoardStructure, vector: np.ndarray, action_index) -> np.array:
    """Generate mask for building a street in the setup phase.

    The street must be free and adjacent to the last built village.

    Args:
        structure (BoardStructure): Board metadata and index maps.
        vector (np.ndarray): Current state vector.
        action_index (int): Action index identifying the village that the
            new street must connect to.

    Returns:
        np.ndarray: Mask for the street-building action (dtype int16).
    """

    # pass is not a possible action, so start with all zeros
    mask= np.zeros(structure.mask_indices['length'], dtype=np.int16)

    # vector representing the build villages
    villages = np.zeros(structure.no_of_nodes, dtype=np.int16)
    villages[structure.index_to_action(action_index)[1]] = 1

    # build options are edges that are not yet occupied AND that are adjacent to the just build village
    build_options_for_streets = np.logical_and(structure.edge_node_matrix  @ villages , vector[structure.vector_indices['edges']] == 0)

    mask[structure.mask_indices['streets']] = build_options_for_streets

    return mask
    

def mask_from_vector_for_building_village(structure: BoardStructure, vector: np.ndarray) -> np.array:
    """Generate mask for building a village in the setup phase.

    The village must be free and adjacent to the last built street.

    Args:
        structure (BoardStructure): Board metadata and index maps.
        vector (np.ndarray): Current state vector.

    Returns:
        np.ndarray: Mask for the village-building action (dtype int16).
    """

    # pass is not a possible action, so start with all zeros
    mask= np.zeros(structure.mask_indices['length'], dtype=np.int16)
    free_nodes_on_board = np.logical_not((vector[structure.vector_indices['nodes']] > 0)  @ structure.node_neighbour_matrix)

    mask[structure.mask_indices['villages']] = free_nodes_on_board
    return mask


def rotate_vector_backward(structure: BoardStructure, vector: np.ndarray) -> np.ndarray:
    """Rotate the state vector backward by one player (shift perspective).

    This shifts player-perspective so that the active player moves one step
    backwards. A copy is returned; the input is not modified.

    Args:
        structure (BoardStructure): Board metadata and index maps.
        vector (np.ndarray): State vector to rotate.

    Returns:
        np.ndarray: Rotated state vector (copy).
    """
    vector = vector.copy()
    vector_indices = structure.vector_indices
    
    # Define transformation via a lookup table
    # Index is the original value, value is the transformed value
    lookup = np.array([0, 2, 3, 4, 1, 6, 7, 8, 5])  # 0→0, 1→2, 2→3, 3→4, 4→1, 5→6, 6→7, 7→8, 8→5

    # rotate ranks
    vector[vector_indices['ranks']] = (vector[vector_indices['ranks']] - 1) % 4

    # Apply transformation only at specified indices
    vector[vector_indices['edges']] = lookup[vector[vector_indices['edges']].astype(int)]
    vector[vector_indices['nodes']] = lookup[vector[vector_indices['nodes']].astype(int)]

    # rotate hands
    temp = vector[vector_indices['hand_for_player'][3]].copy()

    # Perform rotation (in reverse order to avoid overwrite issues)
    vector[vector_indices['hand_for_player'][3]] = vector[vector_indices['hand_for_player'][2]]
    vector[vector_indices['hand_for_player'][2]] = vector[vector_indices['hand_for_player'][1]]
    vector[vector_indices['hand_for_player'][1]] = vector[vector_indices['hand_for_player'][0]]
    vector[vector_indices['hand_for_player'][0]] = temp

    return vector

def rotate_vector_forward(structure: BoardStructure, vector: np.ndarray) -> np.ndarray:
    """Rotate the state vector forward by one player (shift perspective).

    This shifts player-perspective so that the active player moves one step
    forwards. A copy is returned; the input is not modified.

    Args:
        structure (BoardStructure): Board metadata and index maps.
        vector (np.ndarray): State vector to rotate.

    Returns:
        np.ndarray: Rotated state vector (copy).
    """
    vector = vector.copy()
    vector_indices = structure.vector_indices
    
    # Define transformation via a lookup table
    # Index is the original value, value is the transformed value
    lookup = np.array([0, 4, 1, 2, 3, 8, 5, 6, 7])  # 0→0, 1→4, 2→1, 3→2, 4→3, 5→9, 6→5, 7→6, 8→7

    # rotate ranks
    vector[vector_indices['ranks']] = (vector[vector_indices['ranks']] + 1) % 4

    # Apply transformation only at specified indices
    vector[vector_indices['edges']] = lookup[vector[vector_indices['edges']].astype(int)]
    vector[vector_indices['nodes']] = lookup[vector[vector_indices['nodes']].astype(int)]

    # rotate hands
    temp = vector[vector_indices['hand_for_player'][0]].copy()

    # Perform rotation (in reverse order to avoid overwrite issues)
    vector[vector_indices['hand_for_player'][0]] = vector[vector_indices['hand_for_player'][1]].copy()
    vector[vector_indices['hand_for_player'][1]] = vector[vector_indices['hand_for_player'][2]].copy()
    vector[vector_indices['hand_for_player'][2]] = vector[vector_indices['hand_for_player'][3]].copy()
    vector[vector_indices['hand_for_player'][3]] = temp.copy()

    return vector

def vector_throw_dice(structure, vector, enforce: int = -1) -> tuple[int, np.ndarray]:
    """Throw dice and update the vector with resource distributions.

    Args:
        structure (BoardStructure): Board metadata including dice tables
            and impact matrices.
        vector (np.ndarray): State vector to update (modified in place and
            returned).
        enforce (int): If -1 (default) roll randomly; otherwise a number
            to enforce as the dice total (implementation sets the first
            die to this value and the second to zero).

    Returns:
        tuple[int, np.ndarray]: (dice_total, updated vector).
    """
    if enforce == -1:
        dice_1 = np.random.choice([1,2,3,4,5,6])
        dice_2 = np.random.choice([1,2,3,4,5,6])
    else:
        dice_1,dice_2 = enforce, 0
    nodes = vector[structure.vector_indices['nodes']]
    if (dice_1+dice_2) in structure.values:
        dice_value = structure.dice_results.index(dice_1 + dice_2)
        for player_index, hand_indices in enumerate(structure.vector_indices['hand_for_player']):
            added_resources_villages = (nodes == (player_index + 1)) @ structure.dice_impact_per_node_dnt[dice_value]
            added_resources_towns =  2*(nodes == (player_index + 5)) @ structure.dice_impact_per_node_dnt[dice_value]
            vector[hand_indices] += added_resources_villages + added_resources_towns
            
    # when you throw 7 you have to hand in half of your resources
    elif (dice_1+dice_2) == structure.dice_value_to_hand_in_cards:
        for player_index, hand_indices in enumerate(structure.vector_indices['hand_for_player']):
            if sum(vector[hand_indices]) > 7:
                qty_to_remove = sum(vector[hand_indices]) // 2
                for _ in range(qty_to_remove):
                    options = np.nonzero(vector[hand_indices])[0]
                    card_out = np.random.choice(options)
                    vector[hand_indices[card_out]] -= 1

    return (dice_1+dice_2), vector

def calculate_street_length_first_player(structure, vector) -> int:
    """Calculate the longest street length for the first player.

    Args:
        structure (BoardStructure): Board metadata and connectivity matrices.
        vector (np.ndarray): State vector to inspect.

    Returns:
        int: Length (number of connected edges) of the longest street for
            player 0.
    """
    already_build_streets = structure.nodes_connected_as_array[vector[structure.vector_indices['edges']] == 1]
    values, counts = np.unique(already_build_streets, return_counts=True)
    start_or_stop_nodes = values[counts == 1]
    routes = [{'end':start,'used':[]} for start in start_or_stop_nodes]
    longest = 0
    while routes:
        route = routes.pop(0)
        longest = max(len(route['used']),longest)
        for index,edge in enumerate(already_build_streets):
            if index in route['used']  or route['end'] not in edge:
                continue
            if edge[0] == route['end']:
                routes.append({'end':edge[1],'used':route['used'] + [index]})
            else:
                routes.append({'end':edge[0],'used':route['used'] + [index]})
    return longest

def calculate_street_length_all_players(structure, vector) -> np.ndarray:
    """Calculate the longest street length for all players.

    Args:
        structure (BoardStructure): Board metadata and connectivity matrices.
        vector (np.ndarray): State vector to inspect.

    Returns:
        np.ndarray: Array of longest street lengths for each player.
    """
    lengths = []
    for i in range(structure.no_of_players):
        already_build_streets = structure.nodes_connected_as_array[vector[structure.vector_indices['edges']] == i+1]
        values, counts = np.unique(already_build_streets, return_counts=True)
        start_or_stop_nodes = values[counts == 1]
        routes = [{'end':start,'used':[]} for start in start_or_stop_nodes]
        longest = 0
        while routes:
            route = routes.pop(0)
            longest = max(len(route['used']),longest)
            for index,edge in enumerate(already_build_streets):
                if index in route['used']  or route['end'] not in edge:
                    continue
                if edge[0] == route['end']:
                    routes.append({'end':edge[1],'used':route['used'] + [index]})
                else:
                    routes.append({'end':edge[0],'used':route['used'] + [index]})
        lengths.append(longest)
    return np.array(lengths)

def calculate_score_first_player(structure, vector) -> int:
    """Calculate the score for the first player in the vector.

    Args:
        structure (BoardStructure): Board metadata and scoring rules.
        vector (np.ndarray): State vector to inspect.

    Returns:
        int: Calculated score for player 0.
    """             
    street_lengths = calculate_street_length_all_players(structure, vector)
    score = 0
    score += np.count_nonzero(vector[structure.vector_indices['nodes']] == 5) * 2
    score += np.count_nonzero(vector[structure.vector_indices['nodes']] == 1) * 1
    # check if street_lengths[0] is larger than 3 and alone the highest in street_lengths
    score += 2 if street_lengths[0] > 3 and street_lengths[0] > np.max(street_lengths[1:]) else 0
    return int(score)

def calculate_score_all_players(structure, vector) -> np.ndarray:
    """
    Calculate the score for all players in the vector.

    Args:
        structure (BoardStructure): Board metadata and scoring rules.
        vector (np.ndarray): State vector to inspect.

    Returns:
        np.ndarray: Scores for all players (dtype int16).
    """
    street_lengths = calculate_street_length_all_players(structure, vector)
    scores = []
    for i in range(structure.no_of_players):
        score = 0
        score += np.count_nonzero(vector[structure.vector_indices['nodes']] == (i+5)) * 2
        score += np.count_nonzero(vector[structure.vector_indices['nodes']] == (i+1)) * 1
        # check if street_lengths[i] is larger than 3 and alone the highest in street_lengths
        score += 2 if street_lengths[i] > 3 and street_lengths[i] > np.max(np.delete(street_lengths, i)) else 0
        scores.append(score)
    return np.array(scores, dtype=np.int16)


def build_street(structure, vector, action_index) -> np.ndarray:
    """Build a street for the player on the edge with the given index.

    Args:
        structure (BoardStructure): Board metadata including cost tables.
        vector (np.ndarray): Current state vector.
        action_index (int): Action index identifying the edge to build.

    Returns:
        np.ndarray: A new vector (copy) with the street built and costs applied.
    """
    new_vector = vector.copy()
    cost = np.array(structure.real_estate_cost[0])
    indices_for_hand = structure.vector_indices['hand_for_player'][0]
    index_for_edge = structure.vector_indices['edges'][structure.action_parameters[action_index]]
    new_vector[index_for_edge] = 1
    new_vector[indices_for_hand] -= cost
    return new_vector
    
def build_village(structure, vector, action_index) -> np.ndarray:
    """Build a village for the player on the node with the given index.

    Args:
        structure (BoardStructure): Board metadata including cost tables.
        vector (np.ndarray): Current state vector.
        action_index (int): Action index identifying the node to build on.

    Returns:
        np.ndarray: A new vector (copy) with the village built and costs applied.
    """
    new_vector = vector.copy()
    cost = np.array(structure.real_estate_cost[1])
    indices_for_hand = structure.vector_indices['hand_for_player'][0]
    index_for_node = structure.vector_indices['nodes'][structure.action_parameters[action_index]]
    new_vector[index_for_node] = 1
    new_vector[indices_for_hand] -= cost
    return new_vector

def build_town(structure, vector, action_index) -> np.ndarray:
    """Build a town for the player on the node with the given index.

    Args:
        structure (BoardStructure): Board metadata including cost tables.
        vector (np.ndarray): Current state vector.
        action_index (int): Action index identifying the node to upgrade.

    Returns:
        np.ndarray: A new vector (copy) with the town built and costs applied.
    """
    new_vector = vector.copy()
    cost = np.array(structure.real_estate_cost[2])
    indices_for_hand = structure.vector_indices['hand_for_player'][0]
    index_for_node = structure.vector_indices['nodes'][structure.action_parameters[action_index]]
    new_vector[index_for_node] = 5
    new_vector[indices_for_hand] -= cost
    return new_vector

def trade_between_players(structure, vector, action_index, player_accepting_trade: int = None) -> np.ndarray:
    """Perform a resource trade between players.

    The trade parameters are taken from ``structure.action_parameters[action_index]``
    which yields a (card_out, card_in) tuple (indices of resource types).

    Args:
        structure (BoardStructure): Board metadata including action parameters.
        vector (np.ndarray): Current state vector.
        action_index (int): Action index identifying the trade.
        player_accepting_trade (int | None): If provided, the index of the
            player accepting the trade; when None only player 0's hand is
            updated (useful for simulating proposals).

    Returns:
        np.ndarray: A new vector (copy) with updated hands reflecting the trade.
    """
    new_vector = vector.copy()
    # create vector indicating trade
    resources = np.zeros(structure.no_of_resource_types, np.int8)
    card_out_in = structure.action_parameters[action_index]
    resources[card_out_in[0]] = -1
    resources[card_out_in[1]] = 1
    # update for first player
    indices_for_hand_to = structure.vector_indices['hand_for_player'][0]
    new_vector[indices_for_hand_to] += resources

    # if available also update for second player (trading partner)
    if player_accepting_trade is not None:
        indices_for_hand_from = structure.vector_indices['hand_for_player'][player_accepting_trade]
        new_vector[indices_for_hand_from] -= resources
    
    return new_vector

def trade_with_bank(structure, vector, action_index) -> np.ndarray:
    """Trade resources with the bank (4-for-1 standard trade).

    Args:
        structure (BoardStructure): Board metadata including action parameters.
        vector (np.ndarray): Current state vector.
        action_index (int): Action index identifying the bank trade.

    Returns:
        np.ndarray: A new vector (copy) with updated resources after the bank trade.
    """
    new_vector = vector.copy()
    
    resources = np.zeros(structure.no_of_resource_types, np.int8)
    card_out_in = structure.action_parameters[action_index]
    resources[card_out_in[0]] = -4
    resources[card_out_in[1]] = 1
    
    indices_for_hand = structure.vector_indices['hand_for_player'][0]
    new_vector[indices_for_hand] += resources
    return new_vector

def add_to_hand_of_first_player(structure: BoardStructure, vector: np.ndarray, resources: np.ndarray) -> np.ndarray:
    """Add resources to the hand of the first player and return a new vector.

    Args:
        structure (BoardStructure): The board structure.
        vector (np.ndarray): The current state vector.
        resources (np.ndarray): The resources to add.

    Returns:
        np.ndarray: New vector with resources added to player 0's hand.
    """
    new_vector = vector.copy()
    indices_for_hand = structure.vector_indices['hand_for_player'][0]
    new_vector[indices_for_hand] += resources
    return new_vector

def execute_action_on_vector_for_first_player(structure, vector, action_index, trading_partner: int = None):
    """Execute an action (by index) for the first player and return the updated vector.

    Args:
        structure (BoardStructure): Board metadata including action types and parameters.
        vector (np.ndarray): Current state vector.
        action_index (int): Action index to execute.
        trading_partner (int | None): Optional index of trading partner for
            player-to-player trades.

    Returns:
        np.ndarray: New vector with the action applied (copy or modified as per
            the underlying helper functions).

    Raises:
        ValueError: If the action type is unknown.
    """
    action_type = structure.action_types[action_index]
    if action_type == 'street':
        return build_street(structure, vector, action_index)
    elif action_type == 'village':
        return build_village(structure, vector, action_index)
    elif action_type == 'town':
        return build_town(structure, vector, action_index)
    elif action_type == 'trade_player' or action_type == 'trade_specific_player':
        return trade_between_players(structure, vector, action_index, player_accepting_trade=trading_partner)
    elif action_type == 'trade_bank':
        return trade_with_bank(structure, vector, action_index)
    elif action_type == None:
        return vector
    else:
        raise ValueError(f"Unknown action type: {action_type}")   
    

def get_street_indices_for_player(structure: BoardStructure, vector: np.ndarray, player: int) -> np.ndarray:
    """
    Get the indices of the streets built by the given player.

    Args:
        structure (BoardStructure): The board structure.
        vector (np.ndarray): The current state vector.
        player (int): The player index (0-3).

    Returns:
        np.ndarray: The indices of the streets built by the player.
    """
    if player < 0 or player >= structure.no_of_players:
        raise ValueError(f"Player index {player} is out of bounds. Must be between 0 and {structure.no_of_players - 1}.")
    street_value = player + 1
    street_indices = np.where(vector[structure.vector_indices['edges']] == street_value)[0]
    return street_indices

def get_village_indices_for_player(structure: BoardStructure, vector: np.ndarray, player: int) -> np.ndarray:
    """
    Get the indices of the villages built by the given player.

    Args:
        structure (BoardStructure): The board structure.
        vector (np.ndarray): The current state vector.
        player (int): The player index (0-3).

    Returns:
        np.ndarray: The indices of the villages built by the player.
    """
    if player < 0 or player >= structure.no_of_players:
        raise ValueError(f"Player index {player} is out of bounds. Must be between 0 and {structure.no_of_players - 1}.")
    village_value = player + 1
    village_indices = np.where(vector[structure.vector_indices['nodes']] == village_value)[0]
    return village_indices

def get_town_indices_for_player(structure: BoardStructure, vector: np.ndarray, player: int) -> np.ndarray:
    """
    Get the indices of the towns built by the given player.

    Args:
        structure (BoardStructure): The board structure.
        vector (np.ndarray): The current state vector.
        player (int): The player index (0-3).

    Returns:
        np.ndarray: The indices of the towns built by the player.
    """
    if player < 0 or player >= structure.no_of_players:
        raise ValueError(f"Player index {player} is out of bounds. Must be between 0 and {structure.no_of_players - 1}.")
    town_value = player + 5
    town_indices = np.where(vector[structure.vector_indices['nodes']] == town_value)[0]
    return town_indices



    
def generate_distance_between_players(structure: BoardStructure, vector: np.ndarray) -> np.ndarray:
    """
    Use self.structure.edge_to_edge_distance_matrix and the occuped streets of the players in vector indices to
    calculate for each player the closest distance to each other player. We use the occupied streets for this.
    """
    street_indices = [get_street_indices_for_player(structure, vector, player) for player in range(structure.no_of_players)]
    distance_matrix = np.full((structure.no_of_players, structure.no_of_players), np.inf)
    for p1 in range(structure.no_of_players):
        p1_edges = street_indices[p1]
        if len(p1_edges) == 0:
            continue
        for p2 in range(structure.no_of_players):
            if p1 == p2:
                distance_matrix[p1, p2] = 0
                continue
            p2_edges = street_indices[p2]
            if len(p2_edges) == 0:
                continue
            # get the minimum distance between any edge of player 1 and any edge of player 2
            min_dist = np.min(structure.edge_to_edge_distance_matrix[np.ix_(p1_edges, p2_edges)])
            distance_matrix[p1, p2] = min_dist
    return distance_matrix