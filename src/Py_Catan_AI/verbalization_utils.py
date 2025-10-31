"""Utilities to convert game actions and state into human-readable text.

This module provides small helpers to turn action indices and vectorized
game state into concise English sentences or multi-sentence descriptions.
The output is intended for logging, UI tooltips, or as input to a
verbalization model (LLM) for personalized commentary.

Functions
    - action_to_sentence: Single-action -> short sentence.
    - create_message: Full turn summary (scores, street changes, trades).

Notes
    - Most functions expect a `BoardStructure` instance and a state
      `vector` where player 0 is the active player.

Author: Rob Hendriks
Version: 1.0.0
"""

from Py_Catan_AI.board_structure import BoardStructure
from Py_Catan_AI.vector_utils import execute_action_on_vector_for_first_player, calculate_score_all_players, calculate_street_length_all_players, generate_distance_between_players
import numpy as np

def action_to_sentence(structure, action_index, player_name="Player") -> str:
    """Return a short human-readable sentence for a single action.

    Args:
        structure (BoardStructure): Board metadata and action mapping helpers.
        action_index (int): Action index to verbalize (may be -1 for a special
            rejected-trade sentinel).
        player_name (str): Human-friendly player name used in the sentence.

    Returns:
        str: Short descriptive sentence about the action.
    """
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

def create_message(structure: BoardStructure, 
                   vector: np.ndarray, 
                   original_action_index: int, 
                   action_to_execute_index: int, 
                   active_player, 
                   trading_partner: int, names: list ) -> str:
    """Build a multi-sentence summary describing a turn and its effects.

    The returned string contains: active player name, the originally proposed
    action, the action that executed (which may differ), whether a trade was
    accepted (and by whom), changes to pairwise street distances, changes to
    the "longest street" ownership, and score changes for all players.

    Args:
        structure (BoardStructure): Board metadata and helper methods.
        vector (np.ndarray): Current state vector (player 0 is active).
        original_action_index (int): Index of the originally proposed action.
        action_to_execute_index (int): Index of the action that executed.
        active_player (int): Index of the active player in the global ordering.
        trading_partner (int | None): If a trade executed, the accepting player's
            index in the global ordering; otherwise None.
        names (list[str]): Player names in global ordering (length == no_of_players).

    Returns:
        str: A compact multi-sentence description suitable for logging or LLM input.

    Notes:
        - The function rotates `names` so that the active player appears first
          (matching the vector's player-0 convention).
        - The function does not modify the provided `vector`.
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


# def rejected_trade_to_sentence(structure, action_index, player_name="Player"):
#     print('Are we using this function? Seems overkill, can we not remove?')
#     return f"The proposed trade is rejected by all players."

# def generate_message_for_distances_between_players(structure, vector, names):
#     print('Are we using this function? Seems overkill, can we not remove?')
#     distances = generate_distance_between_players(structure, vector)
#     messages = []
#     for p1 in range(structure.no_of_players):
#         for p2 in range(structure.no_of_players):
#             if p1 != p2:
#                 messages.append(f"The shortest street distance between {names[p1]} and {names[p2]} is {distances[p1, p2]}.")
#     return "\n".join(messages)