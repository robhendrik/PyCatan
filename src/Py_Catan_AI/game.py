"""High-level PyCatan game runner and logging utilities.

This module builds on the lower-level `PyCatanGameEnv` environment and
provides a player-oriented game runner suitable for simulations,
human-in-the-loop play, and reinforcement-learning data collection.

Responsibilities
        - Orchestrate player objects and their decision callbacks during a full
            Catan game (initial placement + regular gameplay).
        - Produce structured `GameLog` objects capturing per-step state, actions,
            messages and RL replay buffer entries when players expose `rl_log`.
        - Handle trade negotiation between players using masked responses and
            utilities from `vector_utils`.
        - Provide helpers for persisting logs, creating videos from logs and
            enriching logs with multi-agent natural-language comments.

Primary class
        - PyCatanGame(PyCatanGameEnv): concrete game runner that extends the
            environment with a game loop, player orchestration and logging helpers.

Public API highlights
        - PyCatanGame.play_catan_game(players=None) -> GameLog
        - PyCatanGame.save_game_logs(file_name, game_log)
        - PyCatanGame.add_comments_with_openai(game_log)
        - PyCatanGame.generate_and_save_video(game_log, filename)
        - PyCatanGame.summarize_game_results(game_log)

Usage example
    >>> from Py_Catan_AI.game import PyCatanGame
        >>> env = PyCatanGame()
        >>> game_log = env.play_catan_game()
        >>> print(env.summarize_game_results(game_log))

Notes
        - Player objects passed to `play_catan_game` must implement:
                - decide_best_action(vector, mask) -> int
                - respond_positive_to_other_players_trading_request(vector, reverse_mask) -> bool
            and may optionally expose an `rl_log` attribute used for RL logging.
        - The `add_comments_with_openai` helper calls out to an OpenAI-based
            multi-agent commenter; network access and API credentials are required
            for that functionality.

Author: Rob Hendriks
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from Py_Catan_AI.vector_utils import mask_from_vector_for_responding_to_trade_request, rotate_vector_forward
from Py_Catan_AI.verbalization_utils import create_message
from Py_Catan_AI.game_log import  initialize_game_log, create_log_entry, add_log_entry, save_game_log, save_vector_log, save_mask_log
from Py_Catan_AI.openai_interface_multiagent import add_multiagent_comments_to_game_log
from Py_Catan_AI.plotting_utils import video_from_log, plot_board_positions_with_indices_from_vector
from Py_Catan_AI.default_structure import default_players
from Py_Catan_AI.game_env import PyCatanGameEnv
from Py_Catan_AI.game_log import GameLog
from Py_Catan_AI.rl_game_log import RLReplayBuffer

class PyCatanGame(PyCatanGameEnv):
    def __init__(self, structure=None,max_rounds=51, victory_points_to_win=8):#, names=None, personas=None):
        """Initialize a PyCatanGame instance."""
        super().__init__(structure = structure, max_rounds=max_rounds, victory_points_to_win=victory_points_to_win)#, names=names, personas=personas)
        self.game_log = None
    
    def play_catan_game(self, players: list = None):
        """Run a full Catan game with the provided player objects and return a GameLog.

        This method orchestrates a complete game (initial placement + regular
        gameplay) for four players. It repeatedly queries each active player's
        decision callback for a proposed action, resolves trade negotiation
        (if the action is a trade request), applies the chosen action using the
        environment `step()` method, and records structured log entries for
        every step.

        Key behaviors and side-effects:
        - Initializes per-player `RLReplayBuffer` objects (assigned to
            `player.rl_log`) when players expose an `rl_log` attribute.
        - Handles trade negotiation by constructing reverse masks and calling
            `respond_positive_to_other_players_trading_request` on potential
            trading partners. If no partner accepts, a rejected-trade counter
            controls retries and eventual pass behavior.
        - Logs every step using `GameLog` helpers (`initialize_game_log`,
            `create_log_entry`, `add_log_entry`). The result is stored on
            `self.game_log` and returned.
        - After the game ends, enriches any player RL logs with final ranking
            and victory point information and generates per-player metadata
            (position, name, UID) for replay purposes.

        Args:
                players (list | None): Sequence of 4 player objects implementing
                        the interface expected by the runner:
                                - decide_best_action(vector, mask) -> int
                                - respond_positive_to_other_players_trading_request(vector, reverse_mask) -> bool
                        If None, `default_players` is used. Length must be 4.

        Returns:
                GameLog: A `GameLog` object containing the per-step recorded
                DataFrame and references to the `game` and `players` used.

        Raises:
                ValueError: If `players` is not a list of length 4.

        Notes:
                - The environment enforces a per-player maximum of 5 in-round
                    actions; when exceeded that player will pass (action index 0).
                - `action_to_execute_index` values: -1 means skip (stay with same
                    active player), 0 means pass/end-turn, positive integers select
                    concrete actions defined by the board `structure`.
                - The method mutates `self.game_log` and may mutate player
                    objects (e.g. setting/using `rl_log`).
        """
        game = self
        
        # initialize players
        if players is None:
            players = default_players
        names = [p.name for p in players]
        if len(players) != 4:
            raise ValueError("Number of players must be 4.")
        
        # start game
        vector, mask, reward, terminated, truncated, info = game.reset_game()
        failed_trades = np.ones_like(mask)

        # Initialize logging
        game_log = initialize_game_log(game=game, players=players)

        for p in players:
            if hasattr(p, "rl_log"):
                p.rl_log = RLReplayBuffer()  # reset RL log for new game

        # Run game loop
        while not terminated and not truncated:
            # get best_action from active  if not yet 5 actions in this round
            player = players[info['stage']['active_player']]
            proposed_action_index = player.decide_best_action(vector, mask)
            if hasattr(player, "rl_log"):
                player.update_rl_log_game_information(round = info['rounds'], 
                                                        action_in_round = info['action in round'], 
                                                        score = info['score'][0])


            # decide on actual action to execute
            if info['action in round'] >= 5:
                action_to_execute_index = 0 # pass
            elif game.structure.index_to_action(proposed_action_index)[0] == 'trade_player':
                trade_request_replies = []
                for trading_partner in [1,2,3]:
                    reverse_mask = mask_from_vector_for_responding_to_trade_request(
                        structure=game.structure, 
                        vector=vector, 
                        trading_partner=trading_partner, 
                        proposed_trade_index=proposed_action_index
                    )
                    if sum(reverse_mask) == 1:
                        trade_request_replies.append(False)
                    else:
                        rotated_vector = vector.copy()
                        for _ in range(trading_partner):
                            rotated_vector = rotate_vector_forward(game.structure,  rotated_vector)
                        player_to_respond_to_trade_request = players[(info['stage']['active_player'] + trading_partner) % 4]
                        reply = player_to_respond_to_trade_request.respond_positive_to_other_players_trading_request(rotated_vector, reverse_mask)
                        trade_request_replies.append(reply)
                        if hasattr(player_to_respond_to_trade_request, "rl_log"):
                            player_to_respond_to_trade_request.update_rl_log_game_information(round = info['rounds'], 
                                                                                            action_in_round = info['action in round'], 
                                                                                            score = info['score'][trading_partner])

                idx = next((i for i, flag in enumerate(trade_request_replies) if flag), None)
                if idx is not None:
                    # there is a trading partner accepting the trade
                    trading_partner = idx + 1
                    action_to_execute_index = proposed_action_index
                else:
                    # this trade is declines by all
                    if np.sum(failed_trades == 0)> 5:
                        # pass to next player if already 5 failed trades
                        trading_partner = None
                        action_to_execute_index = 0
                    else:
                        trading_partner = None
                        action_to_execute_index = -1 # -1 means skip action altogether, but stay with this player as active player
                        failed_trades[proposed_action_index] = 0 # add to rejected trades to avoid repeating
            else:
                # best action is not a trade request
                trading_partner = None
                action_to_execute_index = proposed_action_index

            # reset the mask for failed trades if the original action is not a trade request
            if game.structure.index_to_action(proposed_action_index)[0] != 'trade_player':
                failed_trades = np.ones_like(mask)
            
            # create message for logging
            message = create_message(structure=game.structure, 
                                    vector=vector, 
                                    names=names,
                                    original_action_index=proposed_action_index, 
                                    action_to_execute_index=action_to_execute_index, 
                                    active_player=info['stage']['active_player'],
                                    trading_partner=trading_partner)
            # log the game state
            entry = create_log_entry(structure = game.structure, 
                                     names = names, 
                                     vector = vector, 
                                     info = info, 
                                     proposed_action_index = proposed_action_index, 
                                     action_to_execute = action_to_execute_index, 
                                     input_message = message)
            game_log = add_log_entry(game_log, entry)
            
            # Execute the action
            if action_to_execute_index >= 0:
                vector, mask, reward, terminated, truncated, info = game.step(action_to_execute_index, trading_partner)
            
            # filter the actions for next round with rejected trades
            mask = np.logical_and(mask, failed_trades)
            
        # create final message for logging
        message = create_message(structure=game.structure, 
                                vector=vector, 
                                names=names,
                                original_action_index=proposed_action_index, 
                                action_to_execute_index=action_to_execute_index, 
                                active_player=info['stage']['active_player'],
                                trading_partner=trading_partner)
        # log the final game state
        entry = create_log_entry(structure = game.structure, 
                            names = names, 
                            vector = vector, 
                            info = info, 
                            proposed_action_index = proposed_action_index, 
                            action_to_execute = action_to_execute_index, 
                            input_message = message)
        game_log = add_log_entry(game_log, entry)
        self.game_log = game_log

        # --- Check which players have created an RL log for reinforcement learning ---
        ranking = self.score_to_rankings(info['score'])
        for this_players_index, p in enumerate(players):
            if hasattr(p, "rl_log"):
                p.rl_log.to_future_return_from_heuristic_value_scaled_for_game(structure=game.structure)
                p.rl_log.add_final_ranking_and_victory_points(final_ranking = ranking[this_players_index], victory_points = info['score'][this_players_index])
                p.rl_log.add_position_in_game_order(position_in_game = this_players_index)
                p.rl_log.add_player_name(player_name = p.name)
                p.rl_log.add_game_UID()
        
        self.game_log = game_log

        return game_log
    
    def score_to_rankings(self, scores: list) -> list:
        """Convert player scores into ranking positions."""
        scores = np.array(scores)
        # Use pandas rank with method="min" and ascending=False
        return pd.Series(scores).rank(method="min", ascending=False).astype(int).tolist()

    def save_game_logs(self, file_name = "game_log.pkl", game_log: GameLog = None) -> None:
        """ Save the provided GameLog to a file."""
        if game_log is None:
            game_log = self.game_log
        save_game_log(game_log, file_name = file_name)
        return

    def save_vector_and_mask_logs(self,
                                   file_name_vector= 'vector_log.csv', 
                                   file_name_mask = 'mask_log.csv', 
                                   game_log: GameLog = None) -> None:
        """ Save the vector and mask logs from the provided GameLog to CSV files."""
        if game_log is None:
            game_log = self.game_log
        save_vector_log(game_log = game_log, filename = file_name_vector)
        save_mask_log(game_log = game_log, filename= file_name_mask)
        return

    def add_comments_with_openai(self, game_log: GameLog = None) -> GameLog:
        """ Add comments to the game log using OpenAI API. Returns the updated GameLog."""
        if game_log is None:
            game_log = self.game_log
        game_log_with_comments = add_multiagent_comments_to_game_log(game_log = game_log)
        self.game_log = game_log_with_comments
        return game_log_with_comments
    
    def generate_and_save_video(self, 
                                game_log: GameLog = None, 
                                filename: str = "game_progress.mp4") -> None:
        """ Generate and save a video from the provided GameLog."""
        if game_log is None:
            game_log = self.game_log
        video_from_log(game_log = game_log, filename=filename)
        return

    def summarize_game_results(self, game_log: GameLog = None) -> str:
        """ Generate a summary string of the game results from the provided GameLog."""
        if game_log is None:
            game_log = self.game_log
        final_entry = game_log.log.iloc[-1]
        summary = f"Game ended in {final_entry.rounds} rounds. \nFinal scores: \n" + ",\n ".join([f"\t{final_entry.player_names[i]}: {final_entry.score[i]} points" for i in range(len(final_entry.player_names))])
        return summary
    
    def plot_game_position(self, game_log: GameLog = None, entry_index: int = -1) -> None:
        """ Plot the game position at a specific entry in the game log."""
        if game_log is None:
            game_log = self.game_log
        structure = game_log.structure
        game = game_log.game
        names = [p.name for p in game_log.players]
        entry = game_log.log.iloc[entry_index].to_dict()
        # ===== THIS IS NOT EFFICIENT, SHOULD NOT RECREATE INFO HERE =====
        info = {
            'stage': {'active_player': entry['active_player'], 'phase': entry['stage']},
            'rounds': entry['rounds'],
            'action in round': entry['action_in_round'],
            'dice result': entry['dice_result'],
            'terminated': entry['terminated'],
            'truncated': entry['truncated'],
            'street_length': np.array(entry['street_length']),
            'score': np.array(entry['score']),
        }
        # Draw/refresh board
        fig =plot_board_positions_with_indices_from_vector(
            structure = structure, 
            input_vector = entry['vector'], 
            names = names, 
            active_player = info['stage']['active_player'], 
            info = info, 
            fig = None
        )
        #fig.show()
        return