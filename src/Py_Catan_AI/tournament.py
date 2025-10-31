"""Tournament helpers for running many PyCatan games and collecting results.

This module provides a simple Tournament runner that repeatedly plays
PyCatan games (via `Py_Catan_AI.game.PyCatanGame`) for a set of players and
collects aggregated statistics and optional RL logs.

The runner is intentionally small and framework-agnostic: it expects player
objects that implement the minimal player interface used by `PyCatanGame`.

Notes
        - The tournament always assumes 4 players. Passing a list of a different
            length will raise a ValueError.
        - The runner produces three primary outputs: per-game tournament points,
            per-game victory points, and per-game rounds played. Optionally it will
            also return collected RL logs when players expose `rl_log` attributes.

Example
        >>> from Py_Catan_AI.tournament import Tournament
        >>> T = Tournament(no_games_in_tournament=24)
        >>> results = T.tournament_rl_training_data_generation(players, output_type='results_only')

Author:
    Rob Hendriks

Version:
    1.0.0

"""
import sys
import numpy as np
import pandas as pd
sys.path.append("../src")
from Py_Catan_AI.game import PyCatanGame
from Py_Catan_AI.default_structure import default_players
from Py_Catan_AI.game_log import victory_points_from_game_log, rounds_from_game_log
from Py_Catan_AI.rl_game_log import RLReplayBuffer

class Tournament:

    def __init__(self,
                 no_games_in_tournament: int = 24,
                 max_rounds_per_game: int = 50,
                 victory_points_to_win: int = 8,
                 verbose: bool = True):
        """Create a Tournament runner.

        Args:
            no_games_in_tournament (int): Number of games to play in the
                tournament (default 24).
            max_rounds_per_game (int): Max rounds per game before truncation.
            victory_points_to_win (int): Victory points required to end a game.
            verbose (bool): Whether to print per-game summaries during execution.

        Notes:
            - The tournament expects exactly 4 players; a ValueError is raised
              if a different number is provided to the runner.
        """
        self.no_games_in_tournament = no_games_in_tournament
        self.score_table_for_ranking_per_game = [10, 5, 2, 0]
        self.max_rounds_per_game = max_rounds_per_game
        self.victory_points_to_win = victory_points_to_win
        self.verbose = verbose
        self.list_of_orders = self._create_list_of_orders()
        self.list_of_reversed_orders = self._create_list_of_reversed_orders()
        random_indicator = np.random.randint(0,1000)
        self.file_name_for_logging = f"game_logging_{random_indicator}.txt"
 
    def tournament(self, players) -> tuple:
        """Compatibility wrapper: play a tournament and return results.

        This convenience method proxies to
        `tournament_rl_training_data_generation(..., output_type='results_only')`.

        Args:
            players (list): Sequence of 4 player objects.

        Returns:
            tuple: (overall_tournament_points, overall_victory_points, overall_rounds)
        """
        return self.tournament_rl_training_data_generation(players, output_type='results_only')
        

    def tournament_rl_training_data_generation(self, 
                                               players, 
                                               gamma=None, 
                                               start_game_number=0, 
                                               stop_game_number=None, 
                                               fixed_player_order=False,
                                               output_type: str = 'results_only') -> tuple:
        """Play a tournament and collect per-game statistics and optional RL logs.

        The tournament runs games for different player orderings (unless
        `fixed_player_order` is provided) and returns aggregated results.

        Args:
            players (list): List of 4 player instances. Each player may expose
                an `rl_log` attribute (an RLReplayBuffer) to collect RL data.
            gamma (float|None): Legacy argument; not used but accepted for
                backward compatibility.
            start_game_number (int): Index of the first game to play (inclusive).
            stop_game_number (int|None): Index of the last game to play (exclusive).
                If None, plays until `no_games_in_tournament`.
            fixed_player_order (bool|int): If False (default), rotate player
                order across games. If an integer (1-24) is provided, that
                specific permutation is used for all games.
            output_type (str): One of 'full', 'logs_only', 'results_only' controlling
                what is returned.

        Returns:
            tuple|list: When `output_type == 'full'` returns
                (overall_tournament_points, overall_victory_points, overall_rounds, all_rl_logs).
                When `output_type == 'results_only'` returns the first three arrays.
                When `output_type == 'logs_only'` returns only `all_rl_logs`.

        Raises:
            ValueError: If the player list length != 4 or if invalid arguments are
                supplied for `fixed_player_order` or `output_type`.
        """
        if stop_game_number is None:
            start_game_number=0
            stop_game_number = self.no_games_in_tournament
        if stop_game_number - start_game_number > self.no_games_in_tournament:
            raise ValueError("The range of games to play exceeds the total number of games in the tournament.")
        if gamma is not None:
            print("⚠️ gamma argument will not be used in this function. Future returns will have to be added after data generation.")
        if output_type not in ['full', 'logs_only', 'results_only']:
            raise ValueError("output_type must be one of 'full', 'logs_only', or 'results_only'.")
        if isinstance(fixed_player_order, bool) and fixed_player_order is False:
            change_player_order = True
            order_index = None
        elif isinstance(fixed_player_order, int) and (1 <= fixed_player_order <= 24):
            change_player_order = False
            order_index = fixed_player_order - 1  # convert to 0-based index
        else:
            raise ValueError("fixed_player_order must be False or an integer between 1 and 24.")

        # === check and prepare player data ===
        if len(players) != 4:
            raise ValueError("The tournament requires exactly 4 players.")
        rl_log_players = [p for p in players if hasattr(p, "rl_log")]
        if len(rl_log_players) > 1:
            print(f"⚠️ Warning: More than one player with rl_log in tournament. This may lead to mixed data.")
        players = [p.copy() for p in players]  # work on copies to avoid side effects
        player_names_for_tournament = [p.name for p in players]
        
        # == initialize arrays to store tournament results ===
        overall_tournament_points = np.zeros((self.no_games_in_tournament,len(players)),np.float64)
        overall_victory_points = np.zeros((self.no_games_in_tournament,len(players)),np.int16)
        overall_rounds = np.zeros((self.no_games_in_tournament,len(players)),np.int16)

        # === initialize the list to store RL logs if needed ===
        all_rl_logs = []  # collect RL logs across games

        for game_index, game_number in enumerate(range(start_game_number, stop_game_number)):
            # === change order for every game to avoid same player always goes first
            if change_player_order:
                order_index = game_number
  
            players_for_this_game = self._order_elements(order_index, [p.copy() for p in players],reverse = False)

            # These lists seem to be redundant, consider removing one of them
            player_names_for_game = [p.name for p in players_for_this_game]
            name_order_for_this_game = [p.name.strip() for p in players_for_this_game]

            # === Reset RL logs for players that have them
            for p in players_for_this_game:
                if hasattr(p, "rl_log"):
                    p.rl_log = RLReplayBuffer()  # reset RL log for new game

            # === play the game
            this_game = PyCatanGame(max_rounds=self.max_rounds_per_game, victory_points_to_win=self.victory_points_to_win)
            this_game_log  = this_game.play_catan_game(players = players_for_this_game)

            # === get the points per player at final stage of the game, in the 'game order' ===
            name_point_dict = victory_points_from_game_log(this_game_log)
            final_victory_points = [name_point_dict[player_name] for player_name in player_names_for_game]
            assert np.all(final_victory_points == this_game_log.log.iloc[-1].score) # we can do this more efficienct !!

            # === Collect RL logs if available
            for p in players_for_this_game:
                if hasattr(p, "rl_log"):
                    #p.rl_log.finalize_rewards(gamma=gamma)
                    df = p.rl_log.to_dataframe()
                    df["game_indicator"] = game_number
                    all_rl_logs.append(df)

            # === reverse the random order to assign the points to the right player
            # these victory points are now in 'tournament order', so same order as players for the tournament.
            victory_points_for_this_game = np.array(self._order_elements(order_index, final_victory_points, reverse=True))
            assert np.all(victory_points_for_this_game == [name_point_dict[player_name] for player_name in  player_names_for_tournament]) # check if we did the right thing !!
            rounds_for_this_game = np.array(rounds_from_game_log(this_game_log))
            tournament_points_for_this_game = np.array(self._calculate_points(victory_points_for_this_game))
            player_names = self._order_elements(order_index, [player_name for player_name in player_names_for_game], reverse=True)
            assert all([name_1 == name_2 for name_1, name_2 in zip(player_names, player_names_for_tournament)]) # check if we did the right thing !!

            # === add results from this game to the tournament results ===
            overall_tournament_points[game_index] = tournament_points_for_this_game
            overall_victory_points[game_index] = victory_points_for_this_game
            overall_rounds[game_index] = np.full(len(players),rounds_for_this_game,np.int16)
            # === print ===
            if self.verbose:
                print(f'\nResults for game nr. {str(game_number+1)}:')
                print('Order of players for this game: ' + ', '.join(name_order_for_this_game))
                print('\nPlayer      \t\tResults\t\tPoints\t\tRounds')
                for i, p in enumerate(players_for_this_game):
                    print(f"{player_names[i]}\t\t{tournament_points_for_this_game[i]:.2f}\t\t{victory_points_for_this_game[i]:.2f}\t\t{rounds_for_this_game}")

        if output_type == 'full':
            return overall_tournament_points, overall_victory_points, overall_rounds, all_rl_logs
        elif output_type == 'logs_only':
            return all_rl_logs
        elif output_type == 'results_only':
            return overall_tournament_points, overall_victory_points, overall_rounds


    def print_tournament_results(self, overall_tournament_points, overall_victory_points, overall_rounds, players) -> None:
        """Print a human-readable summary of aggregated tournament results.

        Args:
            overall_tournament_points (np.ndarray): Shape (games, players) with
                tournament-point scores.
            overall_victory_points (np.ndarray): Shape (games, players) with
                raw victory points per game.
            overall_rounds (np.ndarray): Shape (games, players) with rounds per game.
            players (list): List of player objects (must expose `.name`).
        """
        print('\nOverall tournament results:')
        print('Player      \t\tAvg Points\tStd Points\tAvg Victory Pts\tStd Victory Pts\tAvg Rounds\tStd Rounds')
        for i, p in enumerate(players):
            avg_points = np.mean(overall_tournament_points[:,i])
            std_points = np.std(overall_tournament_points[:,i])/np.sqrt(self.no_games_in_tournament)
            avg_victory_points = np.mean(overall_victory_points[:,i])
            std_victory_points = np.std(overall_victory_points[:,i])/np.sqrt(self.no_games_in_tournament)
            avg_rounds = np.mean(overall_rounds[:,i])
            std_rounds = np.std(overall_rounds[:,i])/np.sqrt(self.no_games_in_tournament)
            print(f"{p.name}\t\t{avg_points:.2f}\t\t{std_points:.2f}\t\t{avg_victory_points:.2f}\t\t{std_victory_points:.2f}\t\t{avg_rounds:.2f}\t\t{std_rounds:.2f}")
        return
    
    def log_tournament_results_in_dataframe(self, tournament_index: int, overall_tournament_points: list[int], overall_victory_points: list[int], overall_rounds: list[int], players: list, log: pd.DataFrame = None) -> pd.DataFrame:
        """Return a DataFrame row-wise log of aggregated tournament statistics.

        The function either appends rows to an existing DataFrame `log` or
        creates a new one when `log is None`.

        Args:
            tournament_index (int): Identifier for this tournament run.
            overall_tournament_points (np.ndarray): Shape (games, players).
            overall_victory_points (np.ndarray): Shape (games, players).
            overall_rounds (np.ndarray): Shape (games, players).
            players (list): List of player objects (used for `.name`).
            log (pd.DataFrame | None): Optional DataFrame to append to.

        Returns:
            pd.DataFrame: DataFrame containing one row per player with
            aggregated statistics (mean, standard error) for the tournament.
        """
        if log is None:
            header = ['tournament index', 'player', 'Avg Points', 'Std Points', 'Avg Victory Pts', 'Std Victory Pts', 'Avg Rounds', 'Std Rounds']
            log = pd.DataFrame(columns=header)
        
        for i, p in enumerate(players):
            avg_points = float(np.mean(overall_tournament_points[:,i]))
            std_points = np.std(overall_tournament_points[:,i])/np.sqrt(self.no_games_in_tournament)
            avg_victory_points = float(np.mean(overall_victory_points[:,i]))
            std_victory_points = np.std(overall_victory_points[:,i])/np.sqrt(self.no_games_in_tournament)
            avg_rounds = float(np.mean(overall_rounds[:,i]))
            std_rounds = np.std(overall_rounds[:,i])/np.sqrt(self.no_games_in_tournament)
            log.loc[len(log)] = [tournament_index, p.name, avg_points, std_points, avg_victory_points, std_victory_points, avg_rounds, std_rounds]
    
        return log
    
    def _calculate_points(self,results) -> np.ndarray:
        """Compute tournament points from raw victory-point results.

        The function maps per-game victory point rankings into tournament
        points according to `self.score_table_for_ranking_per_game`. Ties are
        handled by awarding the average score for the tied positions.

        Args:
            results (Sequence[int|float]): Per-player victory points for a
                single game (length == number of players).

        Returns:
            np.ndarray: 1D array of tournament points for the given `results`.
        """
        score_table = self.score_table_for_ranking_per_game
        temp_table = score_table.copy()
        temp_results = results.copy()
        points = np.zeros(len(results),np.float64)
        while max(temp_results) > 0:              
            max_value = max(temp_results)
            indices = [i for i, j in enumerate(temp_results) if j == max_value]
            score = sum(temp_table[:len(indices)])/len(indices)
            for i in indices:
                points[i] = score
                temp_results[i] = -1000
            temp_table = temp_table[len(indices):]
        return points



    def _create_list_of_orders(self) -> list:
        """Create all permutations (orders) for four players.

        Returns:
            list[list[int]]: All 24 permutations of indices [0,1,2,3].
        """
        list_of_orders = [[i] for i in range(4)]
        for _ in range(3):
            new_list = []
            for l in list_of_orders:
                for i in range(4):
                    if i not in l:
                        new_list.append(l+[i])
            list_of_orders = new_list
        return list_of_orders
    
    def _create_list_of_reversed_orders(self) -> list:
        """Create reverse-index mappings for each permutation in `list_of_orders`.

        For each permutation `order`, create an array `reverse_order` such that
        `reverse_order[original_index]` gives the position of that player in
        `order`.

        Returns:
            list[list[int]]: Reverse index lists matching `self.list_of_orders`.
        """
        list_of_reversed_orders = []
        for order in self.list_of_orders:
            reverse_order = [0]*len(order)
            for i,j in enumerate(order):
                reverse_order[j] = i
            list_of_reversed_orders.append(reverse_order)
        return list_of_reversed_orders
    
    def _order_elements(self,game_number: int,elements:list,reverse: bool = False) -> list:
        """Return `elements` reordered according to `game_number`.

        Args:
            game_number (int): Index used to select the permutation.
            elements (list): Sequence of items to reorder (length == 4).
            reverse (bool): If False, reorder to play order; if True, map
                elements back to tournament order (reverse mapping).

        Returns:
            list: Reordered elements according to the selected permutation.
        """
        if not reverse:
            order = self.list_of_orders[game_number % len(self.list_of_orders)]
            ordered_elements = [elements[i] for i in order]
            return ordered_elements
        else:
            reverse_order = self.list_of_reversed_orders[game_number % len(self.list_of_reversed_orders)]
            ordered_elements = [elements[i] for i in reverse_order]
            return ordered_elements
 
