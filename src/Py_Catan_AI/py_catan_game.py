
import numpy as np
import pandas as pd
from Py_Catan_AI.vector_utils import mask_from_vector_for_responding_to_trade_request, rotate_vector_forward
from Py_Catan_AI.verbalization_utils import create_message
from Py_Catan_AI.game_log import  initialize_game_log, create_log_entry, add_log_entry, save_game_log, save_vector_log, save_mask_log
from Py_Catan_AI.game_log import add_rl_info_to_log_entry
from Py_Catan_AI.openai_interface_multiagent import add_multiagent_comments_to_game_log
from Py_Catan_AI.plotting_utils import video_from_log, plot_board_positions_with_indices_from_vector_2
from Py_Catan_AI.default_structure import default_players
from Py_Catan_AI.py_catan_game_env import PyCatanGameEnv
from Py_Catan_AI.game_log import GameLog

class PyCatanGame(PyCatanGameEnv):
    def __init__(self, structure=None,max_rounds=51, victory_points_to_win=8):#, names=None, personas=None):
        super().__init__(structure = structure, max_rounds=max_rounds, victory_points_to_win=victory_points_to_win)#, names=names, personas=personas)
        self.game_log = None


    
    def play_catan_game(self, players: list = None):
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
        for p in players:
            if hasattr(p, "rl_log"):
                p.rl_log.finalize_rewards(gamma = 1)
                df_rl = p.rl_log.to_dataframe()
        
        self.game_log = game_log

        return game_log
    
    def save_game_logs(self, file_name = "game_log.pkl", game_log: GameLog = None):
        if game_log is None:
            game_log = self.game_log
        save_game_log(game_log, file_name = file_name)

    def save_vector_and_mask_logs(self, file_name_vector= 'vector_log.csv', file_name_mask = 'mask_log.csv', game_log: GameLog = None):
        if game_log is None:
            game_log = self.game_log
        save_vector_log(game_log = game_log, filename = file_name_vector)
        save_mask_log(game_log = game_log, filename= file_name_mask)

    def add_comments_with_openai(self, game_log: GameLog = None):
        if game_log is None:
            game_log = self.game_log
        game_log_with_comments = add_multiagent_comments_to_game_log(game_log = game_log)
        self.game_log = game_log_with_comments
        return game_log_with_comments
    
    def generate_and_save_video(self, game_log: GameLog = None, filename: str = "game_progress.mp4"):
        if game_log is None:
            game_log = self.game_log
        video_from_log(game_log = game_log, filename=filename)

    def summarize_game_results(self, game_log: GameLog = None):
        if game_log is None:
            game_log = self.game_log
        final_entry = game_log.log.iloc[-1]
        summary = f"Game ended in {final_entry.rounds} rounds. \nFinal scores: \n" + ",\n ".join([f"\t{final_entry.player_names[i]}: {final_entry.score[i]} points" for i in range(len(final_entry.player_names))])
        return summary
    
    def plot_game_position(self, game_log: GameLog = None, entry_index: int = -1):
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
        fig =plot_board_positions_with_indices_from_vector_2(
            structure = structure, 
            input_vector = entry['vector'], 
            names = names, 
            active_player = info['stage']['active_player'], 
            info = info, 
            fig = None
        )
        fig.show()
        return