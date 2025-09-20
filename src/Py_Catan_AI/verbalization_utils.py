from Py_Catan_AI.board_structure import BoardStructure
from Py_Catan_AI.vector_utils import execute_action_on_vector_for_first_player, calculate_score_all_players, calculate_street_length_all_players, generate_distance_between_players
import numpy as np

def action_to_sentence(structure, action_index, player_name="Player"):
    if action_index >= 0:
        action = structure.index_to_action(action_index)
    else:
        action = ('rejected trade', None)
    if action == (None, None):
        return f"{player_name} ends their turn and passes to the next player."
    action_type, details = action
    if action_type == 'street':
        return f"{player_name} constructs a road at edge {details}."
    elif action_type == 'village':
        return f"{player_name} builds a village at node {details}."
    elif action_type == 'town':
        return f"{player_name} upgrades a village to a town at node {details}."
    elif action_type == 'trade_player':
        return f"{player_name} proposes a trade with another player. Card to give: {structure.resource_type_names[details[0]]}, card to receive: {structure.resource_type_names[details[1]]}."
    elif action_type == 'trade_bank':
        return f"{player_name} trades with the bank. Card to give: {structure.resource_type_names[details[0]]}, card to receive: {structure.resource_type_names[details[1]]}"
    elif action_type == 'rejected trade':
        return f"{player_name}'s trade was rejected. Await the next action for this player"
    else:
        return f"{player_name} takes an unknownaction."

def rejected_trade_to_sentence(structure, action_index, player_name="Player"):
    return f"The proposed trade is rejected by all players."

def generate_message_for_distances_between_players(structure, vector, names):
    distances = generate_distance_between_players(structure, vector)
    messages = []
    for p1 in range(structure.no_of_players):
        for p2 in range(structure.no_of_players):
            if p1 != p2:
                messages.append(f"The shortest street distance between {names[p1]} and {names[p2]} is {distances[p1, p2]}.")
    return "\n".join(messages)

def create_message(structure: BoardStructure, vector: np.ndarray, original_action_index: int, action_to_execute_index: int, active_player, trading_partner: int, names: list ):
    """
    Start the message by given the victory point and street length status for each player. Then describe the original 
    action that was proposed, and then the action that was actually executed. if a trade was accepted state who is the 
    trading partner (by name). If trade was rejected state that the trade was rejected by all player. If the action to execute
    is -1 state that this was the fifth rejected trade in this turn, and that therefore the player is forced to pass their turn.
    If a new street was build show the new distance between all players and say if there is a new owner of the longest street.

    Args:
        structure (_type_): _description_
        original_action_index (_type_): _description_
        action_to_execute_index (_type_): _description_
        trading_partner (_type_, optional): _description_. Defaults to None.
        player_name (str, optional): _description_. Defaults to "Player".
    """
    # The active player is always player 0 for the vector, so adjust the names accordingly
    names = [names[(i + active_player) % structure.no_of_players] for i in range(structure.no_of_players)]
    
    original_action_sentence = action_to_sentence(structure, original_action_index, player_name=names[0])
    executed_action_sentence = action_to_sentence(structure, action_to_execute_index, player_name=names[0])
    
    message = f"The active player is {names[0]}. "
    message += f"The original proposed action was: {original_action_sentence} "
    
    if original_action_index != action_to_execute_index:
        if action_to_execute_index == -1:
            message += f"The proposed trade was rejected by all players. So, {names[0]} can take a next action. "
        elif structure.index_to_action(original_action_index)[0] == 'trade_player' and action_to_execute_index == 0:
            message += f"The proposed trade was rejected by all players. This was the fifth rejected trade, therefore, {executed_action_sentence} "
    else:
        if structure.index_to_action(original_action_index)[0] == 'trade_player':
            message += f"The proposed trade was accepted by {names[trading_partner]}. Therefore the trade was executed as proposed. "
        else:
            message += f"The action was executed as proposed. "
 

    # If a new street was built, show the new distances between all players
    if structure.index_to_action(action_to_execute_index)[0] == 'street':
        temp_vector = execute_action_on_vector_for_first_player(structure, vector, action_to_execute_index, trading_partner=None)
        old_distances = generate_distance_between_players(structure, vector)
        new_distances = generate_distance_between_players(structure, temp_vector)
        for p1 in range(structure.no_of_players):
            for p2 in range(structure.no_of_players):
                if p1 != p2 and old_distances[p1, p2] != new_distances[p1, p2]:
                    message += f"The shortest street distance between {names[p1]} and {names[p2]} has changed from {old_distances[p1, p2]} to {new_distances[p1, p2]}. "
   
        old_street_lengths = calculate_street_length_all_players(structure, vector)
        # check if street_lengths[i] is larger than 3 and alone the highest in street_lengths
        i = 0 # active player is always player 0 in the vector
        already_had_longest_street = True if old_street_lengths[0] > 3 and old_street_lengths[0] > np.max(old_street_lengths[1:]) else False
        new_street_lengths = calculate_street_length_all_players(structure, temp_vector)
        now_has_longest_street = True if new_street_lengths[0] > 3 and new_street_lengths[0] > np.max(new_street_lengths[1:]) else False
        if not already_had_longest_street and now_has_longest_street:
            message += f"{names[0]} now has the longest street with a length of {new_street_lengths[0]}. "
        elif already_had_longest_street and not now_has_longest_street:
            message += f"{names[0]} has lost the longest street. "
        elif already_had_longest_street and now_has_longest_street and old_street_lengths[0] != new_street_lengths[0]:
            message += f"{names[0]} still has the longest street, which is now {new_street_lengths[0]} long (previously {old_street_lengths[0]}). "
        elif already_had_longest_street and now_has_longest_street and old_street_lengths[0] == new_street_lengths[0]:
            message += f"{names[0]} still has the longest street, which remains {new_street_lengths[0]} long. "
        elif not already_had_longest_street and not now_has_longest_street and old_street_lengths[0] != new_street_lengths[0]:
            message += f"{names[0]}'s street length has increased to {new_street_lengths[0]}, but this player still does not have the longest street. "
        elif not already_had_longest_street and not now_has_longest_street and old_street_lengths[0] == new_street_lengths[0]:
            message += f"{names[0]}'s street length remains {new_street_lengths[0]}, and this player does not have the longest street. "

    temp_vector = execute_action_on_vector_for_first_player(structure, vector, action_to_execute_index, trading_partner=trading_partner)
    old_scores = calculate_score_all_players(structure, vector)
    new_scores = calculate_score_all_players(structure, temp_vector)
    message += " Before executing the action, the scores were: " + ", ".join(f"{names[i]} has {old_scores[i]} victory points" for i in range(structure.no_of_players)) + ". "
    for i in range(structure.no_of_players):
        if old_scores[i] != new_scores[i]:
            message += f"After the move {names[i]}'s score has changed from {old_scores[i]} to {new_scores[i]}. "

    return message.strip()
