import sys
import numpy as np
import pandas as pd 
sys.path.append("../src")
from Py_Catan_AI.py_catan_game import PyCatanGame
from Py_Catan_AI.default_structure import default_players
from Py_Catan_AI.game_log import victory_points_from_game_log, rounds_from_game_log


class Tournament:
    def __init__(self, no_games_in_tournament: int = 24):
        '''
        Initialize the tournament with players and board structure.
        The tournament always has 4 players.
        '''
        self.no_games_in_tournament = no_games_in_tournament
        self.score_table_for_ranking_per_game = [10, 5, 2, 0]
        self.max_rounds_per_game = 50
        self.victory_points_to_win = 8
        self.verbose = True
        self.logging = False
        self.list_of_orders = self._create_list_of_orders()
        self.list_of_reversed_orders = self._create_list_of_reversed_orders()
        random_indicator = np.random.randint(0,1000)
        self.file_name_for_logging = f"game_logging_{random_indicator}.txt"

    def _create_list_of_orders(self) -> list:
        '''
        Create a list of orders for the players. It is based on 4 players. 
        All permutations will be in the list.

        returns:
            list_of_orders: list of lists, where each list contains the indices of the players in the order they will play.
        '''
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
        '''
        Create a list of reversed orders for the players.
        The list of reversed orders is a list of lists, where each list contains the indices of the players in the order they will play.
        The order is determined by the game number modulo the length of the list of orders.

        NOTE: We first need to create self.list_of_orders before we can create self.list_of_reversed_orders.

        returns:
            list_of_reversed_orders: list of lists, where each list is used to retrieve the original order.
        '''
        list_of_reversed_orders = []
        for order in self.list_of_orders:
            reverse_order = [0]*len(order)
            for i,j in enumerate(order):
                reverse_order[j] = i
            list_of_reversed_orders.append(reverse_order)
        return list_of_reversed_orders


    def _order_elements(self,game_number: int,elements:list,reverse: bool = False) -> list:
        if not reverse:
            order = self.list_of_orders[game_number % len(self.list_of_orders)]
            ordered_elements = [elements[i] for i in order]
            return ordered_elements
        else:
            reverse_order = self.list_of_reversed_orders[game_number % len(self.list_of_reversed_orders)]
            ordered_elements = [elements[i] for i in reverse_order]     
            return ordered_elements
        
    def tournament(self, players) -> tuple:
        '''
        Run a tournament with the given board structure and players.
        Returns the results of the tournament, including total results, victory points, and rounds played.
        The players are expected to be a list of Player objects.
        The board_structure is expected to be a BoardStructure object.
        The players are expected to be a list of Player objects.
        The tournament will run for a fixed number of games, with players playing in different orders.
        The results will be calculated based on the scores of the players.
        The function will return the total results, victory points, and rounds played for each player.
        The players are expected to be a list of Player objects.
        The board_structure is expected to be a BoardStructure object.
        '''
        if len(players) != 4:
            raise ValueError("The tournament requires exactly 4 players.")
        player_names_for_tournament = [p.name for p in players]
        verbose = self.verbose
        overall_tournament_points = np.zeros((self.no_games_in_tournament,len(players)),np.float64)
        overall_victory_points = np.zeros((self.no_games_in_tournament,len(players)),np.int16)
        overall_rounds = np.zeros((self.no_games_in_tournament,len(players)),np.int16)

        for game_number in range(self.no_games_in_tournament):
            # === change order for every game to avoid same player always goes first
            players_for_this_game = self._order_elements(game_number, [p.copy() for p in players],reverse = False)
            player_names_for_game = [p.name for p in players_for_this_game]
            name_order_for_this_game = [p.name.strip() for p in players_for_this_game]

            # === play the game
            this_game = PyCatanGame(max_rounds=self.max_rounds_per_game, victory_points_to_win=self.victory_points_to_win)
            this_game_log  = this_game.play_catan_game(players = players_for_this_game)

            # === get the points per player at final stage of the game, in the 'game order' ===
            name_point_dict = victory_points_from_game_log(this_game_log)
            final_victory_points = [name_point_dict[player_name] for player_name in player_names_for_game]
            assert np.all(final_victory_points == this_game_log.log.iloc[-1].score) # we can do this more efficienct !!

            # === reverse the random order to assign the points to the right player
            # these victory points are now in 'tournament order', so same order as players for the tournament.
            victory_points_for_this_game = np.array(self._order_elements(game_number, final_victory_points, reverse=True))
            assert np.all(victory_points_for_this_game == [name_point_dict[player_name] for player_name in  player_names_for_tournament]) # check if we did the right thing !!
            rounds_for_this_game = np.array(rounds_from_game_log(this_game_log))
            tournament_points_for_this_game = np.array(self.calculate_points(victory_points_for_this_game))
            player_names = self._order_elements(game_number, [player_name for player_name in player_names_for_game], reverse=True)
            assert all([name_1 == name_2 for name_1, name_2 in zip(player_names, player_names_for_tournament)]) # check if we did the right thing !!

            # === add results from this game to the tournament results ===
            overall_tournament_points[game_number] = tournament_points_for_this_game
            overall_victory_points[game_number] = victory_points_for_this_game
            overall_rounds[game_number] = np.full(len(players),rounds_for_this_game,np.int16)
            # === print ===
            if verbose:
                print(f'\nResults for game nr. {str(game_number+1)}:')
                print('Order of players for this game: ' + ', '.join(name_order_for_this_game))
                print('\nPlayer      \t\tResults\t\tPoints\t\tRounds')
                for i, p in enumerate(players_for_this_game):
                    print(f"{player_names[i]}\t\t{tournament_points_for_this_game[i]:.2f}\t\t{victory_points_for_this_game[i]:.2f}\t\t{rounds_for_this_game}")


        return overall_tournament_points, overall_victory_points, overall_rounds

    def print_tournament_results(self, overall_tournament_points, overall_victory_points, overall_rounds, players):
        '''
        Print the results of the tournament in a readable format. Per player we want to plot 
        the average points, average victory points and average rounds played, including standard deviation.
        '''
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

    def log_tournament_results_in_dataframe(self, tournament_index: int, overall_tournament_points: list[int], overall_victory_points: list[int], overall_rounds: list[int], players: list, log: pd.DataFrame = None) -> pd.DataFrame:
        """Log tournament results in a DataFrame.

        Args:
            tournament_index (int): Index of the tournament.
            overall_tournament_points (list[int]): List of overall tournament points.
            overall_victory_points (list[int]): List of overall victory points.
            overall_rounds (list[int]): List of overall rounds.
            players (list): List of players.
            log (pd.DataFrame, optional): Existing log DataFrame. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the logged tournament results.
        """
        if log is None:
            header = ['tournament index', 'player', 'Avg Points', 'Std Points', 'Avg Victory Pts', 'Std Victory Pts', 'Avg Rounds', 'Std Rounds']
            log = pd.DataFrame(columns=header)
        
        for i, p in enumerate(players):
            avg_points = np.mean(overall_tournament_points[:,i])
            std_points = np.std(overall_tournament_points[:,i])/np.sqrt(self.no_games_in_tournament)
            avg_victory_points = np.mean(overall_victory_points[:,i])
            std_victory_points = np.std(overall_victory_points[:,i])/np.sqrt(self.no_games_in_tournament)
            avg_rounds = np.mean(overall_rounds[:,i])
            std_rounds = np.std(overall_rounds[:,i])/np.sqrt(self.no_games_in_tournament)
            log.loc[len(log)] = [tournament_index, p.name, avg_points, std_points, avg_victory_points, std_victory_points, avg_rounds, std_rounds]
    
        return log
    
    def calculate_points(self,results):
        '''
        Calculate the points for each player based on their results.
        The points are calculated based on the score table for ranking per game.    
        if multiple players have the same score, they will receive the average of the scores for those positions.
        '''
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

 
