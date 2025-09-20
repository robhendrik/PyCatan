import re
import os
import time
import json
import random
import pandas as pd
from openai import OpenAI
from Py_Catan_AI.vector_utils import get_street_indices_for_player, get_town_indices_for_player,get_village_indices_for_player,generate_distance_between_players

client = OpenAI(
  api_key=os.environ["OPENAI_API_KEY"]
# assumes OPENAI_API_KEY is set, check with echo %OPENAI_API_KEY%

def clean_plaintext(text):
    return re.sub(r'[^\x20-\x7E]', '', text).strip()

# def add_comments_to_game_log(game_log: pd.DataFrame, game) -> pd.DataFrame:
#     for i in range(len(game_log)):
#         entry = game_log.iloc[i]
#         comment = generate_comments_from_game_log_entry(entry, game_log.iloc[:i], game.personas, game)
#         game_log.at[i, 'comments'] = comment
#         time.sleep(0.2)  # to avoid hitting rate limits
#     return game_log

# def generate_comments_from_game_log_entry(entry, history, personas, game, model="gpt-4o-mini", timeout=20):
#     player_id = entry["active_player"]
#     player_name = entry['player_names'][player_id]
#     move_text = entry["message"] 
#     history,assistance = create_history_from_game_log(pd.DataFrame([entry]), game.structure, last_n=10)
#     system_prompt = f"You are roleplaying {personas[player_id]} in a game of Catan. It takes {game.structure.winning_score} victory points to win. The game will be ended after {game.structure.max_rounds} rounds if no one reaches the winning score. This round is round {entry['rounds']} and you have {entry['score'][player_id]} victory points."
#     user_prompt = f"Recent conversation:\n{history}\n\nNew move: {move_text}\nReply as {player_name}."
#     assistance_prompt = f"Here is some additional information that might help you understand the game state:\n{assistance}\n\nReply as {player_name}."
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "assistant", "content": assistance_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             temperature=0.7, max_tokens=40, timeout=timeout
#         )
#         comment = clean_plaintext(response.choices[0].message.content.strip())
#     except Exception as e:
#         comment = f"[{player_name} has no comment due to error: {e}]"
#     return comment

def create_history_from_game_log(game_log: pd.DataFrame, structure, last_n=3):
    history = []
    assistance = []
    for i in range(max(0, len(game_log) - last_n-1), len(game_log)-1):
        entry = game_log.iloc[i]
        player_id = entry["active_player"]
        player_name = entry['player_names'][player_id]
        move_text = entry["message"]
        comment = entry.get("comments", "")
        round = entry["rounds"]
        score = ",".join([f"{entry['player_names'][idx+1]} has {s} victory points." for idx, s in enumerate(entry["score"])])
        history.append(f"This was round {round}: The active player was {player_name}. The move was {move_text}. The comments was: {comment}. The scores were: {score}. This is some information on how much space there is between players: {generate_message_for_distances_between_players(structure, entry['vector'], names=entry['player_names'])}")
    for i in range(max(0, len(game_log) - last_n-1), len(game_log)-1):
        entry = game_log.iloc[i]
        vector = entry['vector']
        assistance_message = "\nThis was the status in round {round}, action {action_in_round}:".format(
            round=round, action_in_round=entry["action_in_round"])
        for player_index, pname in enumerate(entry['player_names']):
            assistance_message += f"{pname} owned streets on these edges: {get_street_indices_for_player(structure, vector, player_index)}."
        for player_index, pname in enumerate(entry['player_names']):
            assistance_message += f"{pname} owned villages on these nodes: {get_village_indices_for_player(structure, vector, player_index)}."
        for player_index, pname in enumerate(entry['player_names']):
            assistance_message += f"{pname} owned towns on these nodes: {get_town_indices_for_player(structure, vector, player_index)}."
        assistance_message += generate_message_for_distances_between_players(structure, vector, names=entry['player_names'])
        assistance.append(f"This was round {round}: {assistance_message}")
    return "\n".join(history), "\n".join(assistance)

def generate_message_for_distances_between_players(structure, vector, names):
    distances = generate_distance_between_players(structure, vector)
    messages = []
    for p1 in range(structure.no_of_players):
        for p2 in range(structure.no_of_players):
            if p1 != p2:
                messages.append(f"The shortest street distance between {names[p1]} and {names[p2]} is {distances[p1, p2]}.")
    return "\n".join(messages)

# def generate_comments_from_game_log(game_log, structure, personas, model="gpt-4o-mini", timeout=20):
#     print('Warning, function is deprecated')
#     comments = []
#     for i, entry in enumerate(game_log):
#         player_id = entry["info"]["stage"]["active_player"]
#         player_name = f"Player {player_id+1}"
#         move_text = entry.get("message", f"{player_name} takes a turn.")
#         history = "\n".join(comments[-3:])
#         system_prompt = f"You are roleplaying {personas[player_id]} in a game of Catan."
#         user_prompt = f"Recent conversation:\n{history}\n\nNew move: {move_text}\nReply as {player_name}."
#         try:
#             response = client.chat.completions.create(
#                 model=model,
#                 messages=[{"role": "system", "content": system_prompt},
#                           {"role": "user", "content": user_prompt}],
#                 temperature=0.7, max_tokens=40, timeout=timeout
#             )
#             comment = clean_plaintext(response.choices[0].message.content.strip())
#             print(f" → {player_name}: {comment}")
#         except Exception as e:
#             comment = f"[{player_name} has no comment due to error: {e}]"
#         comments.append(comment)
#         time.sleep(0.2)
#     return comments

# def save_comments(comments, filename="ai_comments.json"):
#     with open(filename, "w", encoding="utf-8") as f:
#         json.dump(comments, f, indent=2, ensure_ascii=False)
#     print(f"✅ Saved {len(comments)} comments to {filename}")


# def generate_multiagent_comments_from_game_log_entry(entry, history, personas, game, model="gpt-4o-mini", timeout=20):
#     """
#     Generate comments from all players, not just the active one.
#     Each persona replies in their own voice, with access to memory of the game so far.
#     """
#     move_text = entry["message"]
#     history_text, assistance_text = create_history_from_game_log(pd.DataFrame([entry]), game.structure, last_n=10)

#     comments = {}
#     for player_id, persona in enumerate(personas):
#         player_name = entry['player_names'][player_id]
#         system_prompt = (
#             f"You are roleplaying {persona} in a game of Catan. "
#             f"It takes {game.structure.winning_score} victory points to win. "
#             f"The game will end after {game.structure.max_rounds} rounds if no one wins. "
#             f"This is round {entry['rounds']}. "
#             f"You currently have {entry['score'][player_id]} victory points."
#         )
#         user_prompt = (
#             f"Recent conversation:\n{history_text}\n\n"
#             f"New move: {move_text}\n\n"
#             f"Reply in character as {player_name}, giving your reaction."
#         )
#         assistance_prompt = (
#             f"Here is some extra information that might help:\n{assistance_text}\n\n"
#             f"Reply as {player_name}."
#         )
#         try:
#             response = client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "assistant", "content": assistance_prompt},
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 temperature=0.7,
#                 max_tokens=50,
#                 timeout=timeout,
#             )
#             comment = clean_plaintext(response.choices[0].message.content.strip())
#         except Exception as e:
#             comment = f"[{player_name} has no comment due to error: {e}]"
#         comments[player_name] = comment
#         time.sleep(0.2)  # avoid rate limits
#     return comments


class CatanAgent:
    def __init__(self, name, persona, game, model="gpt-4o-mini"):
        self.name = name
        self.persona = persona
        self.model = model
        self.game = game
        self.chat_history = [
            {
                "role": "system",
                "content": (
                    f"You are roleplaying {persona} in a game of Catan. "
                    f"Stay in character, keep comments short but with personality. "
                    f"You are {name}. Speak in first person."
                )
            }
        ]

    def comment(self, entry, assistance="", timeout=20):
        """Generate a comment from this agent, given the current game entry."""
        move_text = entry["message"]
        actor_name = entry['player_names'][entry['active_player']]
        round_no = entry["rounds"]

        user_prompt = (
            f"It is round {round_no}. "
            f"{actor_name} just played: {move_text}. "
            f"You are {self.name}. React in character."
        )

        length_hint = random.choice([
                "Keep it very short (just a phrase).",
                "Reply in one concise sentence.",
                "Give a slightly longer reply (2–3 sentences)."
            ])

        user_prompt = (
                f"{user_prompt}\n\n{length_hint}"
            )

        if assistance:
            self.chat_history.append({"role": "assistant", "content": assistance})

        self.chat_history.append({"role": "user", "content": user_prompt})

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=self.chat_history,
                temperature=0.7,
            max_tokens=60,
            timeout=timeout,
            )
            comment = clean_plaintext(response.choices[0].message.content.strip())
            self.chat_history.append({"role": "assistant", "content": comment})
        except Exception as e:
            comment = f"[{self.name} has no comment due to error: {e}]"

        return comment

def add_multiagent_comments_to_game_log(game_log: pd.DataFrame, game) -> pd.DataFrame:
    print('This function is deprecated, use CatanAgent class instead.')
    # Initialize one persistent agent per player
    agents = {
        name: CatanAgent(name, persona, game)
        for name, persona in zip(game.names, game.personas)
    }

    for i in range(len(game_log)):
        entry = game_log.iloc[i]

        # Provide structured assistance if desired
        _, assistance = create_history_from_game_log(
            pd.DataFrame([entry]), game.structure, last_n=5
        )

        # Only the active player speaks this round
        active_id = entry["active_player"]
        active_name = entry["player_names"][active_id]

        # That agent generates a comment
        comment = agents[active_name].comment(entry, assistance)

        # Store as string (only one comment per turn)
        game_log.at[i, "comments"] = {active_name: comment}

        time.sleep(0.3)  # throttle requests

    return game_log
