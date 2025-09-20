"""BoardStructure module for Catan game board management and AI integration.

This module provides the core infrastructure for creating, managing, and analyzing
Catan game boards. It handles the complex geometric relationships between nodes,
edges, and tiles in a hexagonal grid layout, and provides essential functionality
for AI training and gameplay.

The module implements board coordinate calculations, neighbor relationships,
action space definitions, and logging capabilities necessary for training
machine learning models on Catan gameplay data.

Classes:
    BoardStructure: Main class for board geometry, relationships, and action spaces.

Examples:
    Basic usage:
        >>> from Py_Catan.BoardLayout import BoardLayout
        >>> layout = BoardLayout()
        >>> board = BoardStructure(layout)
        >>> print(f"Board has {board.no_of_nodes} nodes and {board.no_of_edges} edges")

    Creating action space for AI training:
        >>> board.included_actions = ["street", "village", "town", "trade_player"]
        >>> action_index = board.action_to_index(("street", 5))
        >>> original_action = board.index_to_action(action_index)

Attributes:
    NO_OF_PLAYERS_ON_BOARD (int): Default number of players (4).

Note:
    This module requires NumPy for mathematical operations and Matplotlib for
    plotting functionality. The BoardLayout class must be imported from the
    Py_Catan package.

Author:
    Rob Hendriks

Version:
    1.0.0
"""

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field, asdict
from Py_Catan_AI.board_layout import BoardLayout
    
class BoardStructure:
    """Comprehensive board structure management for Catan game boards.
    
    This class handles all geometric calculations, spatial relationships, and
    action space definitions for Catan game boards. It provides the foundation
    for AI training by converting game states into vector representations and
    managing valid action spaces.
    
    The class calculates coordinates for nodes, edges, and tiles in a hexagonal
    grid, determines neighbor relationships, computes earning potential for
    positions, and manages action-to-index conversions for machine learning.
    
    Attributes:
        values (list): Dice values for each tile on the board.
        tile_layout (list): Resource types for each tile.
        no_of_nodes (int): Total number of intersection points on the board.
        no_of_edges (int): Total number of edge positions on the board.
        no_of_resource_types (int): Number of different resource types.
        neighbour_nodes_for_nodes (dict): Mapping of node adjacencies.
        nodes_connected_by_edge (dict): Node pairs connected by each edge.
        dice_impact_per_node_dnt (np.ndarray): Dice-node-tile impact matrix.
        node_earning_power (np.ndarray): Expected resource earning per node.
        possible_trades (list): All valid resource trade combinations.
        included_actions (list): Action types included in the action space.
        mask_space_header (list): String identifiers for all possible actions.
        vector_indices (dict): Index mappings for board vector representation.
        mask_indices (dict): Index mappings for action mask representation.
        
    Example:
        >>> from Py_Catan.BoardLayout import BoardLayout
        >>> layout = BoardLayout()
        >>> board = BoardStructure(layout)
        >>> 
        >>> # Get board dimensions
        >>> print(f"Nodes: {board.no_of_nodes}, Edges: {board.no_of_edges}")
        >>> 
        >>> # Convert action to index for AI training
        >>> action = ("street", 10)
        >>> index = board.action_to_index(action)
        >>> recovered_action = board.index_to_action(index)
        >>> 
        >>> # Check node neighbors
        >>> neighbors = board.neighbour_nodes_for_nodes[5]
        >>> print(f"Node 5 neighbors: {neighbors}")
    """



    def __init__(self, board_layout: BoardLayout = BoardLayout()) -> None:
        """Initialize the board structure and compute all relationships and matrices.
        
        Creates a complete board structure from a BoardLayout configuration,
        calculating all geometric relationships, neighbor mappings, and preparing
        data structures for AI training and gameplay.
        
        This method performs extensive calculations including:
        - Node, edge, and tile coordinate generation
        - Neighbor relationship mapping
        - Dice impact and earning power calculations
        - Action space and vector space setup
        - Matrix representations for efficient operations
        
        Args:
            board_layout (BoardLayout, optional): Board configuration object
                containing layout specifications, resource types, costs, and
                game parameters. Defaults to BoardLayout() for standard game.
                
        Returns:
            None: This method initializes the instance in-place.
            
        Example:
            >>> # Standard board
            >>> board = BoardStructure()
            >>> 
            >>> # Custom board layout
            >>> custom_layout = BoardLayout()
            >>> custom_layout.rings = 3
            >>> custom_board = BoardStructure(custom_layout)
            >>> print(f"Custom board size: {custom_board.no_of_nodes} nodes")
            
        Note:
            The initialization process is computationally intensive for large
            boards due to the O(n²) neighbor calculations and matrix generations.
        """
        # setting up the board structure based on the layout
        self.values = board_layout.values.copy()
        self.tile_layout =  board_layout.tile_layout
        self._scale = board_layout.scale
        self._rings = board_layout.rings
        self.resource_types = board_layout.resource_types
        self.resource_type_names = board_layout.resource_type_names
        self.street_cost = board_layout.street_cost
        self.village_cost = board_layout.village_cost
        self.town_cost = board_layout.town_cost
        self.development_card_cost = board_layout.development_card_cost
        self.winning_score = board_layout.winning_score
        self.dice_value_to_hand_in_cards = board_layout.dice_value_to_hand_in_cards
        self.max_available_real_estate_per_type = [
            board_layout.max_available_streets,
            board_layout.max_available_villages,
            board_layout.max_available_towns
        ]
        self.longest_street_minimum = board_layout.longest_street_minimum
        self.plot_colors_players = board_layout.plot_colors_players
        self.plot_labels_for_resources = board_layout.plot_labels_for_resources
        self.plot_max_card_in_hand_per_type = board_layout.plot_max_card_in_hand_per_type
        self.no_of_players = board_layout.no_of_players

        # setting up the vectors for the hexagonal grid
        self.vectors = [
                np.array([1.0*self._scale,0],np.float16),
                np.array([0.5*self._scale,-0.5*self._scale*np.sqrt(3)],np.float16),
                np.array([-0.5*self._scale,-0.5*self._scale*np.sqrt(3)],np.float16),
                np.array([-1.0*self._scale,0],np.float16),
                np.array([-0.5*self._scale,0.5*self._scale*np.sqrt(3)],np.float16),
                np.array([0.5*self._scale,0.5*self._scale*np.sqrt(3)],np.float16),
            ]
        
        # calculating the coordinates for nodes, edges and tiles
        # as well as the relationships between them
        self._node_coordinates,self._edge_coordinates,self._tile_coordinates = self._calc_coordinates()
        self.neighbour_nodes_for_tiles = self._calc_neighbour_nodes_for_tiles()
        self.nodes_connected_by_edge = self._calc_nodes_connected_by_edge()
        self.nodes_connected_as_array = np.array([list(edge) for edge in self.nodes_connected_by_edge],dtype=np.int16)
        self.neighbour_nodes_for_nodes = self._calc_neighbour_nodes_for_nodes()
        self.secondary_neighbour_nodes_for_nodes = self._calc_secondary_neighbour_nodes_for_nodes()
        self.dice_results = list(sorted(set(self.values)))
        self.dice_impact_per_node_dnt = self._calc_dice_impact_per_node_dnt()
        self.node_earning_power = self._calc_node_earning_power()
        
        self.no_of_nodes = len(self._node_coordinates)
        self.no_of_edges = len(self._edge_coordinates)
        self.no_of_resource_types = len(self.resource_types)
        self.real_estate_cost = [
            tuple(self.street_cost.count(c) for c in board_layout.resource_types),
            tuple(self.village_cost.count(c) for c in board_layout.resource_types),
            tuple(self.town_cost.count(c) for c in board_layout.resource_types),
            tuple(self.development_card_cost.count(c) for c in board_layout.resource_types)
        ]
        
        # possible trades
        self.possible_trades = [(i, j) for i in range(self.no_of_resource_types) for j in range(self.no_of_resource_types) if i != j]
        self.no_of_trades = len(self.possible_trades)

        self.node_neighbour_matrix = np.array([
            [1 if (c==r or c in self.neighbour_nodes_for_nodes[r]) else 0 for c in range(self.no_of_nodes)]
            for r in range(self.no_of_nodes)],dtype=np.int8)
        
        def helper(r,c):
            a = self.nodes_connected_by_edge[r]
            b = self.nodes_connected_by_edge[c]
            return len(set(a).intersection(b)) > 0 and set(a) != set(b)
        
        self.edge_edge_matrix = np.array([
            [1 if helper(r,c) else 0 for c in range(self.no_of_edges)]
            for r in range(self.no_of_edges)],dtype=np.int8)
        
        self.edge_node_matrix = np.array([
            [1 if n in self.nodes_connected_by_edge[e] else 0 for n in range(self.no_of_nodes)]
            for e in range(self.no_of_edges)],dtype=np.int8)
        self.edge_to_edge_distance_matrix = self.generate_edge_to_edge_distance_matrix()
        # ===== Set paramaters for creating board vector, used for 
        # training Keras model and for logging board status =====
        self.included_actions = ["street", "village", "town", "trade_player", "trade_bank"]
        self.number_of_players_for_logging = 4 # default for creating log to train AI
        self.header = self.logging_header()
        self.mask_space_header = self.create_mask_space_header()
        self.mask_space_length = len(self.mask_space_header)
        self.vector_space_header = self.header
        self.vector_space_length = len(self.vector_space_header)
        self.vector_indices = self.get_vector_indices()


        self.mask_indices = self.get_action_mask_indices()

        self.trade_options_array = self.get_trade_options_array()

        self.action_types = [self.index_to_action(index)[0] for index in range(self.mask_indices['length'])]
        self.action_parameters = [self.index_to_action(index)[1] for index in range(self.mask_indices['length'])]



    def _calc_coordinates(self):
        """Calculate coordinates for all nodes, edges, and tiles on hexagonal board.
        
        Generates precise 2D coordinates for every board element using hexagonal
        geometry. The coordinate system is based on concentric rings around a
        center point, with each ring containing nodes, edges, and tiles arranged
        in hexagonal patterns.
        
        The calculation uses predefined directional vectors to traverse the
        hexagonal grid systematically, ensuring proper spacing and alignment
        of all board elements.
        
        Returns:
            tuple: A 3-tuple containing:
                - node_coordinates (list): (x, y) coordinates for each node
                - edge_coordinates (list): Coordinate pairs for each edge
                - tile_coordinates (list): (x, y) coordinates for each tile center
                
        Note:
            This is a private method called during initialization. Coordinates
            are stored in instance variables for later use in geometry calculations.
            
        Example:
            The method generates coordinates like:
            - Node 0: (0.0, 0.0) for center
            - Node 1: (1.0, 0.0) for first ring
            - Edge coordinates connect adjacent nodes
            - Tile coordinates mark hex centers
        """
        node_coordinates = []
        tile_coordinates = []
        edge_coordinates = []
        for r in range(self._rings):
            nodes_in_ring = []
            i = 0
            nodes_in_ring.append((1+r)*self.vectors[(4+i)%6] + r*self.vectors[(5+i)%6])
            for _ in range(6):
                nodes_in_ring.append(nodes_in_ring[-1] + self.vectors[i%6])
                for _ in range(r):
                    nodes_in_ring.append(nodes_in_ring[-1] + self.vectors[(i+1)%6])
                    nodes_in_ring.append(nodes_in_ring[-1] + self.vectors[i%6])
                i+=1
            nodes_in_ring.pop(-1)
      
            i = 0
            tiles_in_ring = []
            if r==0:
                tiles_in_ring.append(nodes_in_ring[0]+self.vectors[1] )
            else:
                module = 3+2*(r-1) # number of nodes in ring divided by 6
                for index in range(len(nodes_in_ring)):
                    if index % module == 0:
                        i+=1
                        tiles_in_ring.append(nodes_in_ring[index]+self.vectors[(i)%6] )
                        for step in range(1,r):  
                            tiles_in_ring.append(nodes_in_ring[index+step*2]+self.vectors[(i)%6] ) 
  
            edges_in_ring = []
            for node_number,node in enumerate(nodes_in_ring):
                edges_in_ring.append([node, nodes_in_ring[(node_number+1)%len(nodes_in_ring)]])
        
            if r < self._rings-1:
                module = 3+2*(r-1) # number of nodes in ring divided by 6
                i = 0
                for index in range(len(nodes_in_ring)):
                    if index % module == 0:
                        edges_in_ring.append([nodes_in_ring[index],nodes_in_ring[index]+self.vectors[(i+4)%6]] )
                        i+=1
                        for step in range(r):
                            edges_in_ring.append([nodes_in_ring[index+1+step*2],nodes_in_ring[index+1+step*2]+self.vectors[(i+4)%6]] )

            node_coordinates = node_coordinates + nodes_in_ring
            edge_coordinates = edge_coordinates + edges_in_ring
            tile_coordinates = tile_coordinates + tiles_in_ring
        return node_coordinates,edge_coordinates,tile_coordinates
  
    def _calc_nodes_connected_by_edge(self) -> list:
        '''
        Calculates the nodes connected by each edge in the hexagonal grid.
        Each edge connects two nodes, and this method returns a list of sets,       
        where each set contains the indices of the two nodes connected by that edge.
        The edges are calculated based on the number of rings in the hexagonal grid.
        The first ring has 6 edges, the second ring has 12 edges, and so on.
        The number of edges in each ring is given by the formula: 6 * (r + 1) for r >= 0.
        Returns:
            list: A list of sets, where each set contains two node indices connected by an edge.'''
        nodes_connected_by_edge = []
        for r in range(self._rings):
            node_this_ring = 6*r*r # 0,6,24,54 ... #18 +12(r-1) = 12r + 6, som = 6R + 6R(R-1) =6R^2 
            node_next_ring = 6*(r+1)*(r+1)
            for node_number in range(node_this_ring,node_next_ring-1):
                nodes_connected_by_edge.append(set([node_number, node_number+1]))
            nodes_connected_by_edge.append(set([6*r*r,6*(r+1)*(r+1)-1]))
            if r < self._rings-1:
                module = 3+2*(r-1) # number of nodes in ring divided by 6
                for index,node_number in enumerate(range(node_this_ring,node_next_ring)):
                    if index % module == 0:
                        if node_next_ring == 6*(r+1)*(r+1):
                            nodes_connected_by_edge.append(set([node_this_ring,6*(r+2)*(r+2)-1]) )
                            node_this_ring += 1
                            node_next_ring += 2
                        else:
                            nodes_connected_by_edge.append(set([node_this_ring,node_next_ring]) )
                            node_this_ring += 1
                            node_next_ring += 3
                        for step in range(r):
                            nodes_connected_by_edge.append(set([node_this_ring,node_next_ring]) )
                            node_this_ring += 2
                            node_next_ring += 2
        return nodes_connected_by_edge
  
    def _calc_neighbour_nodes_for_nodes(self):
        neighbour_nodes_for_nodes = [set([]) for _ in range(len(self._node_coordinates))]
        for edge in self.nodes_connected_by_edge:
            edge = list(edge)
            neighbour_nodes_for_nodes[edge[0]].add(edge[1])
            neighbour_nodes_for_nodes[edge[1]].add(edge[0])
        return neighbour_nodes_for_nodes

    def _calc_secondary_neighbour_nodes_for_nodes(self):
        secondary_neighbour_nodes_for_nodes = [set([]) for _ in range(len(self._node_coordinates))]
        for node,direct_neighbours in enumerate(self.neighbour_nodes_for_nodes):
            for nb in direct_neighbours:
                for secondary_connection in self.neighbour_nodes_for_nodes[nb]:
                    if secondary_connection not in direct_neighbours and secondary_connection is not node:
                        secondary_neighbour_nodes_for_nodes[node].add(secondary_connection)
        return secondary_neighbour_nodes_for_nodes
    
    def _calc_neighbour_nodes_for_tiles(self):
        neighbour_nodes_for_tiles = []
        for r in range(self._rings):
            neighbour_nodes_per_tile = []
            if r==0:
                neighbour_nodes_per_tile.append( set([n for n in range(6)]) )
            else:
                node_ring_above = 6*r*r # 0,6,24,54 ... #18 +12(r-1) = 12r + 6, som = 6R + 6R(R-1) =6R^2 
                node_ring_below = 6*(r-1)*(r-1)
                for _ in range(6):
                    neighbour_nodes_below = [ node_ring_below + n for n in range(2)]
                    if neighbour_nodes_below[-1] == 6*r*r:
                        neighbour_nodes_below[-1] = 6*(r-1)*(r-1)
                    if node_ring_above == 6*r*r:
                        neighbour_nodes_above =  [node_ring_above + (12*r+6) -1] +  [(node_ring_above + n)  for n in range(0,3)] 
                    else:
                        neighbour_nodes_above =  [(node_ring_above + n)  for n in range(0,4)] 
                    neighbour_nodes_per_tile.append(set(neighbour_nodes_below + neighbour_nodes_above))
                    node_ring_below, node_ring_above =  neighbour_nodes_below[-1],neighbour_nodes_above[-1]
                    for _ in range(1,r):
                        neighbour_nodes_below = [ node_ring_below + n for n in range(3)]
                        neighbour_nodes_above =  [node_ring_above + n for n in range(3)] 
                        node_ring_below, node_ring_above =  neighbour_nodes_below[-1],neighbour_nodes_above[-1]
                        if neighbour_nodes_below[-1] == 6*r*r:
                            neighbour_nodes_below[-1] = 6*(r-1)*(r-1)
                        neighbour_nodes_per_tile.append(set(neighbour_nodes_below + neighbour_nodes_above))
            neighbour_nodes_for_tiles = neighbour_nodes_for_tiles + neighbour_nodes_per_tile
        return neighbour_nodes_for_tiles

    def _calc_dice_impact_per_node_dnt(self) -> np.ndarray  :
        '''
        Calculate the dice impact per node for each resource type.
        The dice impact is calculated by iterating over all tiles and for each tile,
        iterating over all its neighbour nodes.
        The dice impact is stored in a numpy array with shape (no_of_dice_results, no_of_nodes, no_of_resource_types).
        The first dimension corresponds to the dice results (1-6),
        the second dimension corresponds to the nodes, and the third dimension corresponds to the resource types.
        The dice impact is calculated by counting how many times each resource type is present in the neighbour nodes
        for each dice result.
        The method returns the dice impact per node as a numpy array.
        '''
        dice_impact_per_node_dnt = np.zeros((len(self.dice_results),len(self._node_coordinates),len(self.resource_types)),dtype=np.int16)
        for tile in range(len(self._tile_coordinates)):
            resource = self.resource_types.index(self.tile_layout[tile])
            dice_result = self.dice_results.index(self.values[tile])
            for nb in self.neighbour_nodes_for_tiles[tile]:
                dice_impact_per_node_dnt[dice_result,nb,resource] += 1
        return dice_impact_per_node_dnt
    
    def _calc_node_earning_power(self) -> np.ndarray:
        '''
        Calculate the earning power of each node based on the dice results.
        The earning power is calculated by summing the dice impact for each node
        for all possible dice results (1-6) and excluding the result of 7.
        The earning power is stored in a numpy array with shape (no_of_nodes, no_of_resource_types).
        The earning power is calculated by iterating over all nodes and for each node,
        iterating over all possible dice results (1-6) and summing the dice impact  
        for each node and each resource type.
        '''
        node_earning_power = np.zeros((len(self._node_coordinates),len(self.resource_types)),dtype=np.int16)
        for node in range(len(self._node_coordinates)):
            for dice_1 in range(1,7):
                for dice_2 in range(1,7):
                    if dice_1 + dice_2 == 7:
                        continue
                    dice_result = self.dice_results.index(dice_1 + dice_2)
                    node_earning_power[node] += self.dice_impact_per_node_dnt[dice_result,node]
        return node_earning_power
    
    def polar_to_node(self, polar) -> int:
        '''
        Convert polar coordinates to node index.
        The node index is calculated based on the polar coordinates.    
        The formula used is:
        node_index = 3*polar[0]*(polar[0]-1) + polar[1]
        where polar[0] is the ring number and polar[1] is the position in the ring.
        This formula is derived from the fact that each ring has 6 nodes and the nodes are  
        numbered in a clockwise direction starting from the top node of the first ring.
        The first ring has 6 nodes, the second ring has 12 nodes, the third
        ring has 18 nodes, and so on. The node index is calculated by multiplying the ring number
        by 3 and adding the position in the ring.'''
        return 6*polar[0]*polar[0] + polar[1]

    def polar_to_tile(self, polar) -> int:
        '''
        Convert polar coordinates to tile index.
        The tile index is calculated based on the polar coordinates.
        The formula used is:
        tile_index = 3*polar[0]*(polar[0]-1) + polar[1] + 1
        where polar[0] is the ring number and polar[1] is the position in the ring.
        This formula is derived from the fact that each ring has 6 tiles and the tiles are      
        numbered in a clockwise direction starting from the top tile of the first ring.
        The first ring has 6 tiles, the second ring has 12 tiles, the third
        ring has 18 tiles, and so on. The tile index is calculated by multiplying the ring number
        by 3 and adding the position in the ring, and then adding 1 to account for the first tile.'''
        if polar == (0,0):
            return 0
        else:
            return 3*polar[0]*(polar[0]-1) + polar[1] + 1
        
    def polar_to_edge(self, polar) -> int:
        ''' 
        Convert polar coordinates to edge index.
        The edge index is calculated based on the polar coordinates.

        The formula used is:
        edge_index = 6*polar[0]*polar[0] + polar[1] + 6*polar[0] - 1
        where polar[0] is the ring number and polar[1] is the position in the ring.
        This formula is derived from the fact that each ring has 6 edges and the edges are      
        numbered in a clockwise direction starting from the top edge of the first ring.
        The first ring has 6 edges, the second ring has 12 edges, the third
        ring has 18 edges, and so on. The edge index is calculated by multiplying the ring number
        by 6 and adding the position in the ring, and then adding 6 times the 
        ring number minus 1 to account for the edges in the previous rings.

        '''
        #24 + 18(r-1) = 6+18r
        #[6,24,42] ->[6,30,72]
        #6r + 9r(r-1) = 0,6,30,72 = 9*r*r - 3r = 0,6,30
        return 9*polar[0]*polar[0] - 3*polar[0] + polar[1]
 
    def generate_list_of_all_possible_boards(self) -> list:
        ''' 
        Generate a list of all possible board configurations based on the tile layout.
        This method generates all unique combinations of tiles for the first ring,
        ensuring that no tile is repeated in the same position and that the first and last tiles are different.
        It then adds a second row of tiles, ensuring that the same rules apply for the second row as well.
        The method returns a list of all unique board configurations as strings, where each string represents a board configuration.
        The first character of the string is 'D' for desert, followed by the tiles in the first row,
        and then the tiles in the second row.   
        This method is useful for generating all possible board configurations for the game of Catan.

        NOTE: No yet fully using the settings in layout, this is on TODO list. It will only generate boards
        with 2 rings (standard Catan board) and tiles from 'SWGOB' plus desert in the centre.
        '''
        #For first ring number of combinations:
        # sequence of 6 without same twice, including closing circle
        tiles = {'S':4,'W':4,'G':4, 'O': 3, 'B':3}
        boards = [tile for tile in tiles.keys()]
        for _ in range(5):
            new_boards = []
            for board in boards:
                for tile in tiles.keys():
                    if len(board) < 5:
                        if tile != board[-1] and board.count(tile) < tiles[tile]:
                            new_boards.append(board + tile)
                    else:
                        if tile != board[-1] and tile != board[0] and board.count(tile) < tiles[tile]:
                            new_boards.append(board + tile)
            boards = new_boards

        # only unique permutations
        def permutations(s):
            return [s[n:] + s[:n] for n in range(len(s))]

        uniques = []
        for board in boards:
            for p in permutations(board):
                if p in uniques:
                    break
            else:
                uniques.append(board)

        # add second row
        full_boards = []
        for p in uniques:
            for index in range(12):
                if index == 0 and p.count(tile) < tiles[tile]:
                    rings = [tile for tile in tiles if tile != p[0]]
                elif index in [1,3,7,9]:
                    new_rings = []
                    for ring in rings:
                        for tile in tiles:
                            if tile == ring[-1]:
                                continue
                            if tile == p[index//2] or tile == p[(index//2) + 1]:
                                continue
                            if ring.count(tile) + p.count(tile) >= tiles[tile]:
                                continue
                            new_rings.append(ring + tile)
                    rings = new_rings
                elif index in [2,4,6,8]:
                    new_rings = []
                    for ring in rings:
                        for tile in tiles:
                            if tile == ring[-1]:
                                continue
                            if tile == p[index//2]:
                                continue
                            if ring.count(tile) + p.count(tile) >= tiles[tile]:
                                continue
                            new_rings.append(ring + tile)
                    rings = new_rings
                else:
                    new_rings = []
                    for ring in rings:
                        for tile in tiles:
                            if tile == ring[0] or tile == ring[-1]:
                                continue
                            if tile == p[0] or tile == p[-1]:
                                continue
                            if ring.count(tile) + p.count(tile) >= tiles[tile]:
                                continue
                            new_rings.append(ring + tile)
                    rings = new_rings
            for ring in rings:
                full_boards.append( ("D",p,ring))

        return [b[0] + b[1] + b[2] for b in full_boards]
    
    def logging_header(self) -> list[str]:
        """Create comprehensive header for board state logging and AI training data.
        
        Generates column headers for CSV logging that capture the complete game
        state necessary for AI training. The header includes player rankings,
        scores, board occupation status, and resource holdings for all players.
        
        The header structure follows a specific order:
        1. Game metadata (turns remaining, player rankings)
        2. Player scores and values
        3. Node occupation status (which player owns each intersection)
        4. Edge occupation status (which player owns each road)
        5. Player resource hands (cards held by each player)
        
        Returns:
            list[str]: Ordered list of column headers for board state logging.
                Contains headers for:
                - 'turns_before_end': Remaining turns in the game
                - 'rank_A/B/C/D': Current ranking of each player
                - 'value_A/B/C/D': Current victory points of each player
                - 'node_X': Ownership status of node X (0=unoccupied, 1-4=player)
                - 'edge_X': Ownership status of edge X (0=unoccupied, 1-4=player) 
                - 'hand_PLAYER_RESOURCE': Resource count for each player/resource
                
        Example:
            >>> board = BoardStructure()
            >>> headers = board.logging_header()
            >>> print(f"Total columns: {len(headers)}")
            >>> print(f"First few headers: {headers[:5]}")
            >>> # ['turns_before_end', 'rank_A', 'rank_B', 'rank_C', 'rank_D']
            >>> 
            >>> # Use for CSV logging
            >>> import csv
            >>> with open('game_data.csv', 'w') as f:
            >>>     writer = csv.writer(f)
            >>>     writer.writerow(headers)
            
        Note:
            The number of players is determined by `number_of_players_for_logging`
            attribute. Headers are designed to match the output format of
            board state vectorization for AI training consistency.
        """
        headers = [
            'turns_before_end',
            'rank_A',
            'rank_B',
            'rank_C',
            'rank_D',
            'value_A',
            'value_B',
            'value_C',
            'value_D'
        ]
        headers += ['node_'+str(n) for n in range(self.no_of_nodes)]
        headers += ['egde_'+str(n) for n in range(self.no_of_edges)]
        headers += ['hand_A_'+str(n) for n in range(self.no_of_resource_types)]
        headers += ['hand_B_'+str(n) for n in range(self.no_of_resource_types)]
        headers += ['hand_C_'+str(n) for n in range(self.no_of_resource_types)]
        headers += ['hand_D_'+str(n) for n in range(self.no_of_resource_types)]
        return headers
    
    def get_vector_indices(self) -> dict:
        '''
        Create a dictionary with the indices of the attributes in the vector.
        The indices are used to access the attributes in the vector.

        Returns:
            dict: A dictionary with the indices of the attributes in the vector.
        '''
        letter_indicator = [chr(65+i) for i in range(self.no_of_players)]
        indices = dict([])
        count = 0
        indices['turns_before_end'] = count
        count += 1
        for letter in letter_indicator:
            indices[f'rank_{letter}'] = count
            count += 1
        for letter in letter_indicator:
            indices[f'value_{letter}'] = count
            count += 1
        for i in range(self.no_of_nodes):
            indices['node_'+str(i)] = count
            count += 1
        for i in range(self.no_of_edges):
            indices['edge_'+str(i)] = count
            count += 1
        for letter in letter_indicator:
            for i in range(self.no_of_resource_types):
                indices[f'hand_{letter}_{i}'] = count
                count += 1
        indices['length'] = count
        indices['turns'] = [indices['turns_before_end']]
        indices['ranks'] = [indices[f'rank_{letter}'] for letter in letter_indicator]
        indices['values'] = [indices[f'value_{letter}'] for letter in letter_indicator]
        indices['nodes'] = [indices['node_'+str(i)] for i in range(self.no_of_nodes)]
        indices['edges'] = [indices['edge_'+str(i)] for i in range(self.no_of_edges)]
        indices['hands'] = [indices[f'hand_{chr(65+i)}_{j}'] for i in range(self.number_of_players_for_logging) for j in range(self.no_of_resource_types)]
        indices['hand_for_player'] = [[indices[f'hand_{chr(65+i)}_{j}'] for j in range(self.no_of_resource_types)] for i in range(self.no_of_players) ]
        return indices
    


    def get_action_mask_indices(self) -> dict:
        '''
        Get the action mask indices for the current board state.
        Returns:
            dict: A dictionary with the action mask indices.
        '''
        # create indices for the mask space
        indices = {}
        counter = 0
        indices[self._action_to_key((None, None))] = counter
        counter += 1
        if 'street' in self.included_actions:
            for i in range(self.no_of_edges):
                indices[self._action_to_key(('street',i))] = counter
                counter += 1
        if 'village' in self.included_actions:
            for i in range(self.no_of_nodes):
                indices[self._action_to_key(('village',i))] = counter
                counter += 1
        if 'town' in self.included_actions:
            for i in range(self.no_of_nodes):
                indices[self._action_to_key(('town',i))] = counter
                counter += 1
        if 'trade_player' in self.included_actions:
            for i,j in self.possible_trades:
                indices[self._action_to_key(('trade_player',(i,j)))] = counter
                counter += 1
        if 'trade_specific_player' in self.included_actions:
            for p in range(self.no_of_players-1):
                for i,j in self.possible_trades:
                    indices[self._action_to_key(('trade_specific_player',((i,j),p+1)))] = counter
                    counter += 1
        if 'trade_bank' in self.included_actions:
            for i,j in self.possible_trades:
                indices[self._action_to_key(('trade_bank',(i,j)))] = counter
                counter += 1
        indices['length'] = counter

        if 'street' in self.included_actions:
            indices['streets'] = [indices[self._action_to_key(('street',i))] for i in range(self.no_of_edges)]
        if 'village' in self.included_actions:
            indices['villages'] = [indices[self._action_to_key(('village',i))] for i in range(self.no_of_nodes)]
        if 'town' in self.included_actions:
            indices['towns'] = [indices[self._action_to_key(('town',i))] for i in range(self.no_of_nodes)]
        if 'trade_player' in self.included_actions:
            indices['trades_player'] = [indices[self._action_to_key(('trade_player',(i,j)))] for i,j in self.possible_trades]
        if 'trade_specific_player' in self.included_actions:
            indices['trades_specific_player'] = [indices[self._action_to_key(('trade_specific_player',((i,j),p+1)))] for p in range(self.no_of_players-1) for i,j in self.possible_trades]
        if 'trade_bank' in self.included_actions:
            indices['trades_bank'] = [indices[self._action_to_key(('trade_bank',(i,j)))] for i,j in self.possible_trades]
        return indices
    
    def _action_to_key(self,action: tuple) -> str:
        """
        Convert an action tuple to a unique key.

        Args:
            action (tuple): The action tuple to convert.    

        Raises:
            ValueError: If the action tuple is not recognized.

        Returns:
            str: The unique key for the action.
        """
        action_type, action_param = action

        if action_type is None or action_type == 'pass' or action_type == 'none':
            key = 'pass'
        
        elif action_type == 'street':
            if 'street' not in self.included_actions:
                raise ValueError(f"Action type '{action_type}' not included in action space")
            key = f'street_{action_param}'
            
        elif action_type == 'village':
            if 'village' not in self.included_actions:
                raise ValueError(f"Action type '{action_type}' not included in action space")
            key = f'village_{action_param}'
            
        elif action_type == 'town':
            if 'town' not in self.included_actions:
                raise ValueError(f"Action type '{action_type}' not included in action space")
            key = f'town_{action_param}'
            
        elif action_type == 'trade_player':
            if 'trade_player' not in self.included_actions:
                raise ValueError(f"Action type '{action_type}' not included in action space")
            give_resource, get_resource = action_param
            key = f'trade_player_{give_resource}{get_resource}'
            
        elif action_type == 'trade_specific_player':
            if 'trade_specific_player' not in self.included_actions:
                raise ValueError(f"Action type '{action_type}' not included in action space")
            (give_resource, get_resource), player_id = action_param
            key = f'trade_specific_player_p{player_id}_{give_resource}{get_resource}'
            
        elif action_type == 'trade_bank':
            if 'trade_bank' not in self.included_actions:
                raise ValueError(f"Action type '{action_type}' not included in action space")
            give_resource, get_resource = action_param
            key = f'trade_bank_{give_resource}{get_resource}'
            
        else:
            raise ValueError(f"Unknown action type: {action_type}")
        
        return key
    
    def _key_to_action(self, key: str) -> tuple:
        """
        Convert a unique key to an action tuple.

        Args:
            key (str): The unique key to convert.

        Raises:
            ValueError: If the key is not recognized.

        Returns:
            tuple: The action tuple corresponding to the key.
        """
        if key == 'pass':
            action = (None, None)
        else:
            action_type, action_param = self._split_action_key(key)
            if action_type == 'street':
                action = (action_type, int(action_param))
            elif action_type == 'village':
                action = (action_type, int(action_param))
            elif action_type == 'town':
                action = (action_type, int(action_param))
            elif action_type == 'trade_player':
                give_resource, get_resource = int(action_param[0]), int(action_param[1])
                action = (action_type, (give_resource, get_resource))
            elif action_type == 'trade_bank':
                give_resource, get_resource = int(action_param[0]), int(action_param[1])
                action = (action_type, (give_resource, get_resource))
            elif action_type == 'trade_specific_player':
                # action_param = p1_01
                player_id, give_resource, get_resource = int(action_param[1]), int(action_param[3]), int(action_param[4])
                action = (action_type, ((give_resource, get_resource), player_id))
            else:
                raise ValueError(f"Unknown action key: {key}")
        
        return action
    
    def _split_action_key(self,key) -> tuple:
        """
        Split a key into its action type and parameters.

        Args:
            key (str): The key to split.

        Returns:
            tuple: A tuple containing the action type and parameters.
        """
        if key == 'pass':
            return (None, None)
        
        parts = key.split('_')
        
        if parts[0] in ['street', 'village', 'town']:
            return key.split('_', 1)
        
        elif parts[0] == 'trade' and parts[1] == 'player':
            return ('trade_player', parts[2])

        elif parts[0] == 'trade' and parts[1] == 'bank':
            return ('trade_bank', parts[2])

        elif parts[0] == 'trade' and parts[1] == 'specific':
            # trade_specific_player_p1_01
            return ('trade_specific_player', key[-5:])

        else:
            raise ValueError(f"Unknown action key: {key}")
        

    def action_to_index(self, action: tuple) -> int:
        """Convert an action tuple to its corresponding index in the action space.
        
        This method provides the core functionality for AI training by converting
        human-readable action tuples into numerical indices that can be used by
        machine learning models. The action space includes a 'do nothing' action
        at index 0, followed by all valid game actions.
        
        The method supports various action types including building streets,
        villages, and towns, as well as different types of trading actions.
        Each action type has specific parameter formats.
        
        Args:
            action (tuple): Action tuple with format (action_type, parameters):
                - ('street', edge_index): Build street on specified edge
                - ('village', node_index): Build village on specified node  
                - ('town', node_index): Upgrade village to town on node
                - ('trade_player', (give_resource, get_resource)): Trade with any player
                - ('trade_specific_player', ((give, get), player_id)): Trade with specific player
                - ('trade_bank', (give_resource, get_resource)): Trade with bank
                - (None, None): Do nothing action
        
        Returns:
            int: Zero-based index of the action in the action space. Index 0 is
                reserved for the 'do nothing' action, with all other actions
                following sequentially.
        
        Raises:
            ValueError: If the action tuple is not recognized or not included
                in the current action space configuration.
                
        Example:
            >>> board = BoardStructure()
            >>> 
            >>> # Building actions
            >>> street_index = board.action_to_index(('street', 10))
            >>> village_index = board.action_to_index(('village', 5))
            >>> 
            >>> # Trading actions  
            >>> trade_index = board.action_to_index(('trade_player', (1, 2)))
            >>> 
            >>> # Do nothing action
            >>> pass_index = board.action_to_index((None, None))  # Returns 0
            >>> 
            >>> print(f"Street action index: {street_index}")
            
        Note:
            The action must be included in the board's `included_actions` list,
            otherwise a ValueError will be raised. The indexing is consistent
            with the mask space for AI training compatibility.
        """
        key = self._action_to_key(action)
        if key not in self.mask_indices:
            raise ValueError(f"Action {action} not found in action space")
        return self.mask_indices[key]

    def index_to_action(self, index: int) -> tuple:
        """Convert an action space index back to its corresponding action tuple.
        
        This is the inverse operation of action_to_index(), converting numerical
        indices back to human-readable action tuples. This is essential for
        interpreting AI model outputs and converting predictions back to
        game actions.
        
        Args:
            index (int): Zero-based index in the action space. Must be within
                the valid range [0, action_space_size). Index 0 corresponds
                to the 'do nothing' action.
        
        Returns:
            tuple: Action tuple in format (action_type, parameters):
                - (None, None): Do nothing action (index 0)
                - ('street', edge_index): Build street action
                - ('village', node_index): Build village action
                - ('town', node_index): Build town action
                - ('trade_player', (give, get)): Player trade action
                - ('trade_specific_player', ((give, get), player_id)): Specific player trade
                - ('trade_bank', (give, get)): Bank trade action
        
        Raises:
            ValueError: If index is outside the valid range or does not
                correspond to a valid action in the current action space.
                
        Example:
            >>> board = BoardStructure()
            >>> 
            >>> # Convert index back to action
            >>> action = board.index_to_action(15)
            >>> print(f"Action at index 15: {action}")
            >>> 
            >>> # Round-trip conversion
            >>> original_action = ('street', 8)
            >>> index = board.action_to_index(original_action)
            >>> recovered_action = board.index_to_action(index)
            >>> assert original_action == recovered_action
            >>> 
            >>> # Do nothing action is always at index 0
            >>> do_nothing = board.index_to_action(0)
            >>> assert do_nothing == (None, None)
            
        Note:
            This method provides the inverse mapping to action_to_index() and
            is crucial for AI model interpretation and game action execution.
        """
        action_key = self.mask_space_header[index]
        action = self._key_to_action(action_key)
        return action


    def get_trade_options_array(self) -> np.array:
        """
        Get the trade options array for the player.

        Returns:
            np.array: The trade options array.
        """
        # turn a list of tuples into a 2D np array
        trade_options = np.zeros((len(self.possible_trades), len(self.possible_trades[0])), dtype=np.int8)
        for index, trade in enumerate(self.possible_trades):
            resource_in, resource_out = trade
            trade_options[index] = [resource_in, resource_out]
        return trade_options
    
    def create_mask_space_header(self):
        # create one overall header for the actions in included actions
        pass_header = [self._action_to_key((None, None))]  # do nothing action
        self.mask_space_header = pass_header
        if 'street' in self.included_actions:
            street_header = [self._action_to_key(('street',i)) for i in range(self.no_of_edges)]
            self.mask_space_header += street_header
        if 'village' in self.included_actions:
            village_header = [self._action_to_key(('village',i)) for i in range(self.no_of_nodes)]
            self.mask_space_header += village_header
        if 'town' in self.included_actions:
            town_header = [self._action_to_key(('town',i)) for i in range(self.no_of_nodes)]
            self.mask_space_header += town_header
        if 'trade_player' in self.included_actions:
            trade_player_header = [self._action_to_key(('trade_player',(i,j))) for i,j in self.possible_trades]
            self.mask_space_header += trade_player_header
        if 'trade_specific_player' in self.included_actions:
            trade_specific_player_header = [self._action_to_key(('trade_specific_player',((i,j),p+1))) for p in range(self.no_of_players-1) for i,j in self.possible_trades]
            self.mask_space_header += trade_specific_player_header
        if 'trade_bank' in self.included_actions:
            trade_bank_header = [self._action_to_key(('trade_bank',(i,j))) for i,j in self.possible_trades]
            self.mask_space_header += trade_bank_header
        return self.mask_space_header
    


    def generate_edge_to_edge_distance_matrix(self) -> np.ndarray:
        """
        Generate a matrix of size self.no_of_edges x self.no_of_edges representing the distance between each pair of edges. The distance 
        is the number of edges traversed in the shortest path between the two edges. So if edges connect the distance is zero, 
        if they are one edge apart the distance is 1, etc. The output matrix should be a numpy array with dtype np.int8.
        """
        # self.edge_edge_matrix is the adjacency matrix of the edges
        # Algorithm idea
        # Let A = edge_edge_matrix (shape (E, E)).
        # A^k (matrix power) tells you which edges are connected by a walk of length k.
        # You can loop k = 1...E-1, and whenever (A^k)[i, j] > 0 and distance[i, j] not yet set, assign distance = k.

        E = self.edge_edge_matrix.shape[0]
        dist = np.full((E, E), fill_value=-1, dtype=np.int16)  # -1 = not yet reached
        np.fill_diagonal(dist, 0)

        # Current adjacency power
        A_power = self.edge_edge_matrix.copy()

        filled = np.count_nonzero(dist != -1)  # how many entries are already set
        total = E * E

        for k in range(1, E):  # in worst case, max distance ≤ E-1
            # Find all pairs connected by walk of length k
            reachable = (A_power > 0)

            # Find which pairs are newly reached
            newly_reached = (dist == -1) & reachable
            if np.any(newly_reached):
                dist[newly_reached] = k
                filled += np.count_nonzero(newly_reached)

            # Stop if all pairs have a distance
            if filled == total:
                break

            # Prepare next power for paths of length k+1
            A_power = A_power @ self.edge_edge_matrix

        return dist