"""Heuristic value functions for PyCatan.

This module exposes lightweight scoring functions used by baseline players
and decision utilities. Functions operate either on the raw `vector` (from
the perspective where player 0 is active) or on higher-level player-like
objects. The heuristics prefer buildable hands, direct possessions (streets,
villages, towns), earning power and secondary/tertiary build options.

Public functions
    - calculate_value_hand_first_player_to_optimize_for_building_something
    - calculate_value_for_first_player
    - value_for_player_check (older convenience adapter, deprecated)

Notes
    - All functions return a numeric score (float). Higher is better.
    - The `structure` argument must be a `BoardStructure` instance that
      provides index maps and cost/earning tables.

Author: Rob Hendriks
Version: 1.0.0
"""

import numpy as np
from Py_Catan_AI.value_preferences import optimized_1_with_0_for_full_score
from Py_Catan_AI.board_structure import BoardStructure

def calculate_value_hand_first_player_to_optimize_for_building_something(structure: BoardStructure,
                                                                        vector: np.ndarray) -> float:
    """Score the first player's hand with a bias toward buildable hands.

    This heuristic inspects only the resource cards in player 0's hand and
    awards higher scores for hands that already meet build costs (street,
    village, town) or are close to doing so. It ignores board possessions
    (streets/villages/towns) and earning power.

    Args:
        structure (BoardStructure): Board metadata and cost tables.
        vector (np.ndarray): State vector rotated so player 0 is active.

    Returns:
        float: Heuristic score for the hand (higher is better).

    Notes:
        - This function does not modify the input `vector`.
        - The score is unbounded and intended for relative comparisons only.
    """

    hand_for_calculation = vector[structure.vector_indices['hand_for_player'][0]]
    value = np.all( hand_for_calculation >= structure.real_estate_cost[0]) * 10
    value += np.all( hand_for_calculation >= structure.real_estate_cost[1]) * 20
    value += np.all( hand_for_calculation >= structure.real_estate_cost[2]) * 30
    # value of secondary options
    helper = ( hand_for_calculation- np.array(structure.real_estate_cost[0]))
    value += (1 if -1 == np.sum(helper[helper < 0]) else 0) * 5
    helper = ( hand_for_calculation - np.array(structure.real_estate_cost[1]))
    value += (1 if -1 == np.sum(helper[helper < 0]) else 0)  * 11
    helper = ( hand_for_calculation - np.array(structure.real_estate_cost[2]))
    value += (1 if -1 == np.sum(helper[helper < 0]) else 0)  * 15
    # value of tertiary options
    helper = ( hand_for_calculation - np.array(structure.real_estate_cost[1]))
    value += (1 if -2 == np.sum(helper[helper < 0]) else 0) * 2
    helper = ( hand_for_calculation - np.array(structure.real_estate_cost[2]))
    value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * 3

    return value


def calculate_value_for_first_player(structure: BoardStructure, vector: np.ndarray) -> float:
    """Calculate a full heuristic value for player 0 from the state vector.

    The heuristic combines direct possessions (streets/villages/towns),
    cards in hand (weighted by resource preference), earning power from
    occupied nodes, and direct/secondary/tertiary build options.

    Args:
        structure (BoardStructure): Board metadata, index maps and tables.
        vector (np.ndarray): State vector rotated so player 0 is active.

    Returns:
        float: Composite heuristic score for player 0.

    Notes:
        - Does not mutate the provided `vector`.
        - Values are heuristic and tuned via `value_preferences`.
    """
    free_nodes_on_board = np.logical_not((vector[structure.vector_indices['nodes']] > 0)  @ structure.node_neighbour_matrix)
    free_edges_on_board = vector[structure.vector_indices['edges']] == 0
    build_options_for_villages = np.logical_and(free_nodes_on_board, (vector[structure.vector_indices['edges']] == 1) @ structure.edge_node_matrix)
    build_options_for_streets = np.logical_and(free_edges_on_board, (vector[structure.vector_indices['edges']] == 1) @ structure.edge_edge_matrix)

    helper = np.logical_and(np.logical_not(build_options_for_villages),build_options_for_streets @ structure.edge_node_matrix)
    secondary_village_options= np.logical_and(free_nodes_on_board,helper)

    preference = optimized_1_with_0_for_full_score
    
    # initialize value
    value = 0
    
    # value of direct posessions
    value += np.sum(vector[structure.vector_indices['edges']] == 1) * preference.streets
    value += np.sum(vector[structure.vector_indices['nodes']] == 1) * preference.villages
    value += np.sum(vector[structure.vector_indices['nodes']] == 5) * preference.towns

    # value of cards in hand and penalty for too many cards
    penalty_factor = (sum(vector[structure.vector_indices['hand_for_player'][0]])/(sum(vector[structure.vector_indices['hand_for_player'][0]])+preference.penalty_reference_for_too_many_cards) )
    value += penalty_factor * np.inner(vector[structure.vector_indices['hand_for_player'][0]],preference.resource_type_weight) * preference.cards_in_hand

    # value of current earning power
    earning_power = np.sum(structure.node_earning_power[vector[structure.vector_indices['nodes']] == 1],axis=0) + 2*np.sum(structure.node_earning_power[vector[structure.vector_indices['nodes']] == 5],axis=0)
    value += np.dot(earning_power ,preference.resource_type_weight) * preference.cards_earning_power

    # value of direct options
    value += np.all(vector[structure.vector_indices['hand_for_player'][0]] >= structure.real_estate_cost[0]) * preference.hand_for_street
    value += np.all(vector[structure.vector_indices['hand_for_player'][0]] >= structure.real_estate_cost[1]) * preference.hand_for_village
    value += np.all(vector[structure.vector_indices['hand_for_player'][0]] >= structure.real_estate_cost[2]) * preference.hand_for_town

    value += np.sum(build_options_for_streets) * preference.street_build_options
    value += np.sum(build_options_for_villages) * preference.village_build_options

    # value of earning power for direct options
    extra_villages=build_options_for_villages
    secondary_earning_power =  np.sum(structure.node_earning_power[extra_villages == 1],axis=0)
    value += np.dot(secondary_earning_power ,preference.resource_type_weight) * preference.direct_options_earning_power

    # value of secondary options
    helper = (vector[structure.vector_indices['hand_for_player'][0]] - np.array(structure.real_estate_cost[0]))
    value += (1 if -2 == np.sum(helper[helper < 0]) else 0) * preference.hand_for_street_missing_one
    helper = (vector[structure.vector_indices['hand_for_player'][0]] - np.array(structure.real_estate_cost[1]))
    value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * preference.hand_for_village_missing_one
    helper = (vector[structure.vector_indices['hand_for_player'][0]] - np.array(structure.real_estate_cost[2]))
    value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * preference.hand_for_town_missing_one

    value += np.sum(secondary_village_options) * preference.secondary_village_build_options

    # value of secondary options earning power
    extra_villages=secondary_village_options
    secondary_earning_power =  np.sum(structure.node_earning_power[extra_villages == 1],axis=0)
    value += np.dot(secondary_earning_power ,preference.resource_type_weight) * preference.secondary_options_earning_power

    # value of tertiary options
    helper = (vector[structure.vector_indices['hand_for_player'][0]] - np.array(structure.real_estate_cost[1]))
    value += (1 if -2 == np.sum(helper[helper < 0]) else 0) * preference.hand_for_village_missing_two
    helper = (vector[structure.vector_indices['hand_for_player'][0]] - np.array(structure.real_estate_cost[2]))
    value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * preference.hand_for_town_missing_two

    return value


def value_for_player_check(structure: BoardStructure, player: object) -> float:
    """Compute a heuristic score for a player-like object (adapter).

    This adapter accepts a `player` object (with attributes like `streets`,
    `villages`, `towns`, `hand`, and `build_options`) and computes a score
    equivalent to `calculate_value_for_first_player`. It is retained for
    backward compatibility; prefer `calculate_value_for_first_player` when
    working with raw state vectors.

    Args:
        structure (BoardStructure): Board metadata and tables.
        player (object): Player-like object with expected attributes
            (streets, villages, towns, hand, build_options).

    Returns:
        float: Heuristic score for the provided player object.

    Notes:
        - This adapter is kept for backward compatibility; prefer
          `calculate_value_for_first_player` when working with raw state vectors.
    """
    print('WARNING: This function is deprecated. Use calculate_value_for_first_player instead')
    preference = optimized_1_with_0_for_full_score

    # initialize value
    value = 0
    
    # value of direct posessions
    value += np.sum(player.streets) * preference.streets
    value += np.sum(player.villages) * preference.villages
    value += np.sum(player.towns) * preference.towns

    # value of cards in hand and penalty for too many cards
    penalty_factor = (sum(player.hand)/(sum(player.hand)+preference.penalty_reference_for_too_many_cards) )
    value += penalty_factor * np.inner(player.hand,preference.resource_type_weight) * preference.cards_in_hand

    # value of current earning power
    earning_power = np.sum(structure.node_earning_power[player.villages == 1],axis=0) + 2*np.sum(structure.node_earning_power[player.towns == 1],axis=0)
    value += np.dot(earning_power ,preference.resource_type_weight) * preference.cards_earning_power

    # value of direct options
    value += np.all(player.hand >= structure.real_estate_cost[0]) * preference.hand_for_street
    value += np.all(player.hand >= structure.real_estate_cost[1]) * preference.hand_for_village
    value += np.all(player.hand >= structure.real_estate_cost[2]) * preference.hand_for_town

    value += np.sum(player.build_options['street_options']) * preference.street_build_options
    value += np.sum(player.build_options['village_options']) * preference.village_build_options

    # value of earning power for direct options
    extra_villages=player.build_options['village_options']
    secondary_earning_power =  np.sum(structure.node_earning_power[extra_villages == 1],axis=0)
    value += np.dot(secondary_earning_power ,preference.resource_type_weight) * preference.direct_options_earning_power

    # value of secondary options
    helper = (player.hand - np.array(structure.real_estate_cost[0]))
    value += (1 if -2 == np.sum(helper[helper < 0]) else 0) * preference.hand_for_street_missing_one
    helper = (player.hand - np.array(structure.real_estate_cost[1]))
    value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * preference.hand_for_village_missing_one
    helper = (player.hand - np.array(structure.real_estate_cost[2]))
    value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * preference.hand_for_town_missing_one

    value += np.sum(player.build_options['secondary_village_options']) * preference.secondary_village_build_options

    # value of secondary options earning power
    extra_villages=player.build_options['secondary_village_options']
    secondary_earning_power =  np.sum(structure.node_earning_power[extra_villages == 1],axis=0)
    value += np.dot(secondary_earning_power ,preference.resource_type_weight) * preference.secondary_options_earning_power

    # value of tertiary options
    helper = (player.hand - np.array(structure.real_estate_cost[1]))
    value += (1 if -2 == np.sum(helper[helper < 0]) else 0) * preference.hand_for_village_missing_two
    helper = (player.hand - np.array(structure.real_estate_cost[2]))
    value += (1 if -2 == np.sum(helper[helper < 0]) else 0)  * preference.hand_for_town_missing_two

    return value