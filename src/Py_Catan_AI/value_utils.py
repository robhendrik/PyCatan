
import numpy as np
from Py_Catan_AI.value_preferences import optimized_1_with_0_for_full_score




def calculate_value_hand_first_player_to_optimize_for_building_something(structure, vector) -> float:
    '''
    Calculate the value of the first player's hand to incentivize trading for a hand that allows building.
    '''

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


def calculate_value_for_first_player(structure, vector) -> float:
        """
        Calculate the value for a specific player based on their current state and preferences.

        Args:
            structure (BoardStructure): The game structure.
            vector (np.ndarray): The current game state vector.

        Returns:
            float: The calculated value for the player.
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


def value_for_player_check(structure, player: any) -> float:
    """
    Calculate the value for a specific player based on their current state and preferences. The
    player object is expected to have attributes such as streets, villages, towns, hand, and build_options.

    Args:
        player (Player): The player for whom the value is calculated.

    Returns:
        float: The calculated value for the player.
    """
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