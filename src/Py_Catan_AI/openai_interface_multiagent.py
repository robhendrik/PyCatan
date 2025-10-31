"""OpenAI multi-agent interface for PyCatan.

This module provides helpers to synthesize short, in-character comments from
multiple simulated agents using OpenAI chat models. It converts recent game
log rows into compact prompts, invokes the OpenAI chat completions API and
stores back short comments per-agent.

Expectations & environment:
    - The environment variable ``OPENAI_API_KEY`` must be set before creating
        the OpenAI client (the module reads it at import time).
    - The code assumes a `GameLog` wrapper where each row's ``info``/``vector``
        fields contain the full game-state needed by helper utilities in
        `Py_Catan_AI.vector_utils`.

Public utilities:
    - `create_history_from_game_log` / `extract_recent_comments` – convert a
        `GameLog` DataFrame into compact prompt text for the model.
    - `add_multiagent_comments_to_game_log` – run agents over the log and
        attach short, persona-driven comments to each row's `comments` field.

Note: This module performs network calls. In unit tests you may want to
mock `client.chat.completions.create` or construct `CatanAgent` instances
with a fake client.
"""

import re
import os
import time
import json
import random
import pandas as pd
from openai import OpenAI
from Py_Catan_AI.vector_utils import (
    get_street_indices_for_player,
    get_town_indices_for_player,
    get_village_indices_for_player,
    generate_distance_between_players,
)
from Py_Catan_AI.game_log import GameLog

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # expects OPENAI_API_KEY


def clean_plaintext(text):
    """Strip non-printable ASCII characters and trim whitespace.

    Args:
        text (str): Raw input text that may contain non-ASCII or control
            characters from model output.

    Returns:
        str: A cleaned, printable string suitable for storing in game logs.
    """
    return re.sub(r'[^\x20-\x7E]', '', text).strip()


def generate_message_for_distances_between_players(structure, vector, names):
    """Create human-readable distance lines between every pair of players.

    Args:
        structure: Board structure object providing topology and player count.
        vector (np.ndarray): State vector used to compute distances.
        names (list[str]): Player names in game order.

    Returns:
        str: Newline-separated sentences describing pairwise shortest
            street distances (one line per ordered pair).
    """
    distances = generate_distance_between_players(structure, vector)
    messages = []
    for p1 in range(structure.no_of_players):
        for p2 in range(structure.no_of_players):
            if p1 != p2:
                messages.append(
                    f"The shortest street distance between {names[p1]} and {names[p2]} is {distances[p1, p2]}."
                )
    return "\n".join(messages)


def create_history_from_game_log(game_log: pd.DataFrame, structure, last_n=3):
    """Generate short human-readable history and assistance strings from a
    `GameLog` DataFrame.

    The function returns two strings: a compact turn-by-turn history and a
    more verbose assistance text that enumerates per-player holdings (streets,
    villages, towns) for the selected recent turns.

    Args:
        game_log (pd.DataFrame): DataFrame of game log rows (as produced by
            `GameLog.log`).
        structure: Board structure required by vector helpers.
        last_n (int): Number of recent turns to include.

    Returns:
        tuple(str, str): (history_text, assistance_text)
    """
    history = []
    assistance = []

    def _comment_to_text(entry):
        c = entry.get("comments", "")
        if isinstance(c, dict):
            parts = [f"{k}: {v}" for k, v in c.items() if v]
            return "; ".join(parts)
        return (f"{entry['player_names'][entry['active_player']]}: {c}"
                if isinstance(c, str) and c else "")

    start = max(0, len(game_log) - last_n - 1)
    end = max(0, len(game_log) - 1)
    for i in range(start, end):
        entry = game_log.iloc[i]
        player_id = entry["active_player"]
        player_name = entry['player_names'][player_id]
        move_text = entry["message"]
        comment_text = _comment_to_text(entry)
        round_no = entry["rounds"]
        score = ", ".join(
            f"{entry['player_names'][idx]} has {s} victory points."
            for idx, s in enumerate(entry["score"])
        )
        history.append(
            "This was round {r}: active player {p}. Move: {m}. "
            "Comments: {c}. Scores: {s}. Distances: {d}".format(
                r=round_no, p=player_name, m=move_text, c=comment_text, s=score,
                d=generate_message_for_distances_between_players(
                    structure, entry['vector'], names=entry['player_names'])
            )
        )

    for i in range(start, end):
        entry = game_log.iloc[i]
        vector = entry['vector']
        round_no = entry["rounds"]
        assistance_message = "\nThis was the status in round {round}, action {action_in_round}:".format(
            round=round_no, action_in_round=entry["action_in_round"])
        for player_index, pname in enumerate(entry['player_names']):
            assistance_message += f"{pname} owned streets on these edges: {get_street_indices_for_player(structure, vector, player_index)}."
        for player_index, pname in enumerate(entry['player_names']):
            assistance_message += f"{pname} owned villages on these nodes: {get_village_indices_for_player(structure, vector, player_index)}."
        for player_index, pname in enumerate(entry['player_names']):
            assistance_message += f"{pname} owned towns on these nodes: {get_town_indices_for_player(structure, vector, player_index)}."
        assistance_message += generate_message_for_distances_between_players(structure, vector, names=entry['player_names'])
        assistance.append(f"This was round {round_no}: {assistance_message}")

    return "\n".join(history), "\n".join(assistance)


def extract_recent_comments(past_df: pd.DataFrame, max_comments: int = 8) -> str:
    """Collect recent human/AI comments from a `GameLog` DataFrame.

    Args:
        past_df (pd.DataFrame): Game log dataframe.
        max_comments (int): Maximum number of comment lines to return.

    Returns:
        str: Newline-separated recent comments (up to `max_comments`).
    """
    lines = []
    if len(past_df) == 0:
        return ""
    for _, e in past_df.tail(max_comments * 2).iterrows():
        c = e.get("comments", "")
        if isinstance(c, dict):
            for name, txt in c.items():
                if txt:
                    lines.append(f"{name}: {txt}")
        elif isinstance(c, str) and c:
            speaker = e['player_names'][e['active_player']]
            lines.append(f"{speaker}: {c}")
    return "\n".join(lines[-max_comments:])


class CatanAgent:
    """A lightweight persona-driven agent that produces short in-character
    comments for Catan game log entries using an OpenAI chat model.

    Instances wrap a `game` reference and a persona description. The
    `comment` method builds a short prompt from a log entry and returns a
    cleaned, printable string.
    """

    def __init__(self, name, persona, game, model="gpt-4o-mini"):
        self.name = name
        self.persona = persona
        self.model = model
        self.game = game
        """Create a CatanAgent.

        Args:
            name (str): Display name for the agent.
            persona (str): Short personality description used in the system
                prompt to guide tone and style.
            game: Reference to the running game object (used for metadata).
            model (str): OpenAI chat model name to use.
        """

        self.system_prompt = {
            "role": "system",
            "content": 
                f"You are roleplaying {persona} in a game of Catan. "
                f"You are {name}. Speak in first person. "
                "Always make sure your tone matches your persona in content and style. "
                "Avoid repeating stock phrases."
                "Game rules (summary): The goal is to reach {self.structure.winning_score} victory points. "
                "You earn points by building settlements (1), upgrading to cities (2), "
                "having the longest road (2). "
                "The board has resources: wood, brick, sheep, wheat, ore. "
                "Players build roads, villages, and towns on their turn."
            }
        

    def comment(self, entry, game_log, timeout=20):
        """Produce a short, persona-driven comment for the given log entry.

        The method constructs a system prompt (from the agent persona), a
        small user prompt describing the active player's recent move, and a
        few assistant-context messages derived from previous log rows. It
        calls the OpenAI chat completion endpoint and returns a cleaned
        plaintext reply.

        Args:
            entry (pd.Series): Single row from `GameLog.log` describing the
                current decision point.
            game_log (pd.DataFrame): Full game log DataFrame (used to build
                assistant context).
            timeout (int): Request timeout in seconds for the OpenAI call.

        Returns:
            str: Short cleaned comment produced by the model.
        """

        # generate user prompt
        move_text = entry["message"]
        round_no = entry["rounds"]
        user_prompt = f"You are the ACTIVE player this turn. Round {round_no}. "
        user_prompt += f"You ({self.name}) just are about to make the move: {move_text}. "
        user_prompt += "React in character. Make sure the tone reflects your persona."
        user_prompt += random.choice([
            "Keep it very short (a word or a phrase), but stay in character.",
            "Reply in one (very) short sentence, but stay in character.",
            "Give a slightly longer thought (one or two sentences), but stay in character."
        ])

        # generate assistance prompts
        lines = extract_past_messages_and_comments(game_log, max_messages=10)
        assistance_prompts = [{"role": "assistant", "content": line} for line in lines]


        overall_prompt_message = [self.system_prompt] + assistance_prompts + [{"role": "user", "content": user_prompt}]

        response = client.chat.completions.create(
            model=self.model,
            messages=overall_prompt_message,
            temperature=0.8,
            max_tokens=70,
            timeout=timeout,
        )
        comment = clean_plaintext(response.choices[0].message.content.strip())

        return comment

def extract_past_messages_and_comments(game_log: pd.DataFrame, max_messages: int = 10) -> list:
    """
    Extract the message and the comment for the past 10 messages (not the last one in the dataframe,
    that we call the current message). The messages and comments have to be returned in a readable
    string stating how many turns ago the message/comment comes from, and then clearly indicates the 
    message and comment
        
    Args:
        game_log (pd.DataFrame): The game log dataframe.
        max_messages (int, optional): The maximum number of messages to extract. Defaults to 10.

    Returns:
        list: A list of strings, each string containing the message and comment from a past turn.
    """
    lines = []
    if len(game_log) <= 1:
        return []
    
    # first get all the previous comments from the active player
    active_entry = game_log.iloc[-1]
    active_player_name = active_entry['player_names'][active_entry['active_player']]
    for j in range(0, max(0, len(game_log) - max_messages - 1)):
        entry = game_log.iloc[j]
        speaker = entry['player_names'][entry['active_player']]
        if speaker == active_player_name:
            comment_text = entry.get("comments", "")
            if isinstance(comment_text, dict):
                comment_text = "; ".join(f"{k}: {v}" for k, v in comment_text.items() if v)
            elif not isinstance(comment_text, str):
                comment_text = ""
            round = entry["rounds"]
            lines.append(f"In round {round} your ({speaker}) comment was: {comment_text}.")

    # for the last max_messages, get all comments
    for i in range(max(0, len(game_log) - max_messages - 1), len(game_log) - 1):
        entry = game_log.iloc[i]
        turn_ago = len(game_log) - 2 - i + 1
        move_text = entry["message"]
        comment_text = entry.get("comments", "")
        if isinstance(comment_text, dict):
            comment_text = "; ".join(f"{k}: {v}" for k, v in comment_text.items() if v)
        elif not isinstance(comment_text, str):
            comment_text = ""
        speaker = entry['player_names'][entry['active_player']]
        if comment_text:
            lines.append(f"{turn_ago} turns ago, {speaker}'s action was: {move_text} and they commented: {comment_text}.")
        else:
            lines.append(f"{turn_ago} turns ago, {speaker}'s action was: {move_text} and they had no comment.")
    return lines

def add_multiagent_comments_to_game_log(game_log: GameLog) -> pd.DataFrame:
    """Run persona agents over a `GameLog` and attach short comments.

    The function constructs a `CatanAgent` for each player (using the
    player's `.name` and `.persona` attributes), iterates the recorded log
    rows and calls each agent to produce a short comment for the active
    player at that row. Comments are written back into the DataFrame's
    ``comments`` column as a single-entry dict mapping ``{player_name: text}``.

    Args:
        game_log (GameLog): GameLog wrapper containing the DataFrame and
            player metadata.

    Returns:
        GameLog: The same wrapper with the ``comments`` column populated.
    """

    agents = {
        p.name: CatanAgent(p.name, p.persona, game_log.game)
        for p in game_log.players
    }

    for i, (idx, entry) in enumerate(game_log.log.iterrows()):
        active_id = entry["active_player"]
        active_name = entry["player_names"][active_id]

        comment = agents[active_name].comment(entry, game_log.log)

        # Write back using the true index label
        game_log.log.at[idx, "comments"] = {active_name: comment}

        time.sleep(0.2)
        #print('*', i, '/', len(game_log.log), " ", active_name, "commented:", comment)

    return game_log

def save_comments(comments, filename="ai_comments.json"):
    """Persist a comments dict to a JSON file using UTF-8.

    Args:
        comments (dict): Mapping of row indices or player names to comment text.
        filename (str): Output filename.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(comments, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(comments)} comments to {filename}")
