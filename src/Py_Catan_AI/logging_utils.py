import numpy as np
import json
import parquet

def action_to_sentence(structure, action_index, player_name="Player"):
    print('Warning, function is deprecated')
    try:
        action = structure.index_to_action(action_index)
        return f"{player_name} performs action: {action}"
    except Exception:
        return f"{player_name} takes an action."
    
def rejected_trade_to_sentence(structure, action_index, player_name="Player"):
    print('Warning, function is deprecated')
    action = structure.index_to_action(action_index)
     # Resource names for trading
    resource_names = ['brick', 'desert', 'grain', 'ore', 'sheep', 'wood']

    give = resource_names[action[1][0]]
    receive = resource_names[action[1][1]]

    return f"{player_name} proposes to trade with another player: gives one of {give}, receives one of {receive}. Trade rejected by all players."

def log_game_state(names, vector, info, action_index, input_message, game, game_log=None):
    print('Warning, function is deprecated, use create_log_entry and save_game_log in game_log.py instead')
    if game_log is None:
        game_log = []
    game_log.append({
        'vector': vector.copy(),
        'info': info.copy(),
        'action_index': action_index,
        'message': action_to_sentence(game.structure, action_index, names[info['stage']['active_player']]) + '\n' + input_message
    })
    return game_log

def serialize_entry(entry):
    print('Warning, function is deprecated, use create_log_entry and save_game_log in game_log.py instead')
    serializable = {}
    for k, v in entry.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, dict):
            serializable[k] = serialize_entry(v)
        elif isinstance(v, list):
            serializable[k] = [serialize_entry(x) if isinstance(x, dict) else
                               (x.tolist() if isinstance(x, np.ndarray) else x)
                               for x in v]
        elif isinstance(v, (int, float, str, type(None))):
            serializable[k] = v
        elif isinstance(v, np.integer):
            serializable[k] = int(v)
        elif isinstance(v, np.floating):
            serializable[k] = float(v)
        else:
            serializable[k] = v
    return serializable

def save_game_log(game_log, filename="game_log.json"):
    print('warning, function is deprecated, use save_game_log in game_log.py instead')
    serializable_log = [serialize_entry(entry) for entry in game_log]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(serializable_log, f, indent=2, ensure_ascii=False)
    print(f"✅ Game log saved to {filename}")
