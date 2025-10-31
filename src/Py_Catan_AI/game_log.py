"""Game logging utilities for PyCatan.

This module defines small data containers and helpers to record a game's
progress in a pandas DataFrame, convert between vector/player orderings,
and persist logs to disk. Entries are stored as `GameLogEntry` instances
and collected inside a `GameLog` wrapper.

Intended usage:
    game_log = initialize_game_log(game, players)
    entry = create_log_entry(...)
    game_log = add_log_entry(game_log, entry)

The module includes convenience exporters for vectors/masks and helpers to
read victory points and rounds from the latest entry.

Note about RL metadata storage:
    The legacy helper `add_rl_info_to_log_entry` was removed. If a
    reinforcement-learning workflow needs to attach policy or value
    predictions to a particular log row, store them inside the
    `GameLogEntry.info` dict using the keys ``'policy_probs'`` and
    ``'state_value'``. Example safe pattern to augment an existing row:

        info = game_log.log.at[row_index, 'info'] or {}
        new_info = {**info, 'policy_probs': policy_probs, 'state_value': value}
        game_log.log.at[row_index, 'info'] = new_info

    This keeps RL metadata inline with other per-decision information and
    avoids needing a separate, mutable column. The DataFrame stores the
    dict for the row; callers are responsible for any serialization when
    persisting to disk.

Author: Rob Hendriks
Version: 1.0.1
"""

from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from Py_Catan_AI.vector_utils import rotate_vector_backward

@dataclass
class GameLogEntry:
    """Single recorded snapshot (one row) in the game log.

    Fields mirror the game state at a single decision point. The `vector`
    field is stored in game-order (player ordering matching `player_names`),
    not the rotated "active-player-first" format used by the environment.
    """
    stage: str
    active_player: int
    rounds: int
    action_in_round: int
    dice_result: int
    terminated: bool
    truncated: bool
    street_length: list
    score: list
    player_names: list
    vector: np.ndarray
    mask: np.ndarray
    proposed_action_index: int
    action_index_to_execute: str
    info: dict
    message: str
    comments: str

@dataclass
class GameLog:
    """Container for the running game log and related metadata.

    Attributes:
        log (pd.DataFrame): DataFrame of GameLogEntry rows (as dicts).
        structure: Board structure used by the game (BoardStructure).
        game: Reference to the running game object.
        players (list): Player instances or descriptors used in the game.
    """
    log: pd.DataFrame
    structure: object
    game: any
    players: list

def initialize_game_log(game, players):
    """
    Initialize an empty game log by creating an empty dataframe with the appropriate columns.

    Returns:
        pd.DataFrame: An empty dataframe with the columns matching the GameLogEntry dataclass.
    """
    columns = GameLogEntry.__annotations__.keys()
    return GameLog(log = pd.DataFrame(columns=columns), structure=game.structure, game=game, players=players)

def add_log_entry(game_log: GameLog, entry: GameLogEntry):
    """Append a `GameLogEntry` to the `game_log` DataFrame.

    This stores a dict-converted row at the end of the DataFrame and
    returns the same `GameLog` wrapper for convenience.

    Args:
        game_log (GameLog): The running game log container.
        entry (GameLogEntry): Entry to append.

    Returns:
        GameLog: The same `game_log` instance (with the row appended).
    """
    game_log.log.loc[len(game_log.log)] = asdict(entry)
    return game_log

def create_log_entry(structure,
                    names, 
                    vector, 
                    info, 
                    proposed_action_index, 
                    action_to_execute,
                    input_message):
    """Create a normalized `GameLogEntry` from environment info.

    The environment represents state vectors with the active player at
    index 0. For easier human consumption the log stores vectors and
    per-player arrays in game order (player indices matching the `names`
    list). This function rotates scores, street lengths and the vector
    into that game order and populates a `GameLogEntry`.

    Args:
        structure (BoardStructure): Board metadata used for rotations.
        names (list[str]): Player names in global ordering.
        vector (np.ndarray): Current state vector (active-player-first).
        info (dict): Extra information produced by the environment (must
            contain keys like 'stage', 'street_length', 'score', etc.).
        proposed_action_index (int): Index of the action originally proposed.
        action_to_execute (int): Index of the action that executed.
        input_message (str): Optional human-readable message describing
            the action (verbalization).

    Returns:
        GameLogEntry: Prepared log entry with rotated fields.
    """
    # info is in the vector-order, every the active player is at index 0 for score and has street index 1 etc.
    # log entry we want to have game-order, where the order of the names is leading for the order of scores etc.
    # we also want to adjust the vector to make sure the same player always has street 1 etc.
    # so teh vector entry in entry['vector'] is different from the vector entry in entry['info']['vector'], only
    # if active player is 0 are these the same.

    # rotate street length and score to game order
    rotated_street_length = np.zeros(len(names), np.int16)
    for i in range(len(names)):
        rotated_street_length[(info['stage']['active_player'] + i) % len(names)] = info['street_length'][i]

    rotated_score = np.zeros(len(names), np.int16)
    for i in range(len(names)):
        rotated_score[(info['stage']['active_player'] + i) % len(names)] = info['score'][i]

    # rotate mask if available (can be dummey operation)
    if 'mask' in info: 
        rotated_mask = info['mask'].copy() # mask is invariant under rotation, always ok for active player. We leave it since if we include trade partners in mask we have to rotate
    # quick check on consistency:
    if not np.all(vector == info['vector']):
        print("Warning! Inconsistent vector in info and passed vector for creating log entry. Use passed vector for further processing.")
    # rotate vector to game order
    rotated_vector = vector.copy()
    for _ in range(info['stage']['active_player']):
        rotated_vector = rotate_vector_backward(structure, rotated_vector)
    # create the log entry
    entry = GameLogEntry(
        stage=info['stage']['phase'], 
        active_player=info['stage']['active_player'],
        rounds=info['rounds'],
        action_in_round=info['action in round'],
        dice_result=info['dice result'],
        terminated=info['terminated'],
        truncated=info['truncated'],
        street_length=rotated_street_length, # has to be rotated to game order
        score=rotated_score, # has to be rotated to game order
        player_names=names.copy(),
        vector=rotated_vector, # has to be rotated to game order
        mask=None if 'mask' not in info else rotated_mask, # mask is invariant under rotation, always ok for active player. 
                                        # we leave it since if we include trade partners in mask we have to rotate
        proposed_action_index=proposed_action_index,
        action_index_to_execute=action_to_execute,
        info = info.copy(),
        message=input_message,
        comments=dict([])
    )
    return entry

def save_game_log(game_log: GameLog, filename="game_log.pkl"):
    """Persist the game log DataFrame to a pickle file.

    Args:
        game_log (GameLog): Log container to persist.
        filename (str): Output filename (pickle format).
    """
    game_log.log.to_pickle(filename)


def load_game_log(game, players,filename="game_log.pkl"):
    """Load a pickled game log and return a `GameLog` wrapper.

    Args:
        game: Running game instance used to populate the wrapper's
            metadata (structure is taken from game.structure).
        players (list): Player list to attach to the wrapper.
        filename (str): Path to the pickle file.

    Returns:
        GameLog: Wrapper containing the loaded DataFrame.
    """
    return GameLog(log = pd.read_pickle(filename), structure=game.structure, game=game, players=players)


def save_vector_log(game_log: GameLog, filename="vector_log.csv"):
    """
    Save the vector log to a file as csv from a dataframe with the vector header and column labels.

    Args:
        game_log (GameLog): The game log containing the vector log.
    """
    vector_columns = game_log.structure.vector_space_header
    vectors = pd.DataFrame(game_log.log['vector'].tolist(), columns=vector_columns)
    # create dataframe with vector_columns as columns and vectors as rows
    vectors.to_csv(filename, index=False)
    return

def save_mask_log(game_log: GameLog, filename="mask_log.csv"):
    """
    Save the mask log to a file as csv from a dataframe with the mask header and column labels.

    Args:
        game_log (GameLog): The game log containing the mask log.
    """
    mask_columns = game_log.structure.mask_space_header
    masks = pd.DataFrame(game_log.log['mask'].tolist(), columns=mask_columns)
    # create dataframe with mask_columns as columns and masks as rows
    masks.to_csv(filename, index=False)
    return  

def victory_points_from_game_log_entry(game_log_entry: GameLogEntry):
    """Return a dict mapping player name -> victory points for an entry.

    Args:
        game_log_entry (GameLogEntry): Single log entry to extract VP from.

    Returns:
        dict: Mapping of player name to integer victory points.
    """
    # The active player is always player 0 for the vector, so adjust the names accordingly
    return {game_log_entry.player_names[i]: game_log_entry.score[i] for i in range(len(game_log_entry.player_names))}

def victory_points_from_game_log(game_log: GameLog):
    """Return victory points mapping from the latest log entry.

    Args:
        game_log (GameLog): Log wrapper from which to extract the last entry.

    Returns:
        dict: Mapping player name -> victory points for the last recorded state.
    """
    return victory_points_from_game_log_entry(game_log.log.iloc[-1])

def rounds_from_game_log(game_log: GameLog):
    """Return the rounds counter from the latest log entry.

    Args:
        game_log (GameLog): Log wrapper to inspect.

    Returns:
        int: Value of the 'rounds' field from the last entry.
    """
    return game_log.log.iloc[-1]['rounds']

# (Legacy `add_rl_info_to_log_entry` removed. See module docstring for
# recommended pattern to store RL metadata in `GameLogEntry.info`.)
