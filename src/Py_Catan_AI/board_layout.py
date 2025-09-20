"""BoardLayout module for Catan game board configuration management.

This module provides the BoardLayout dataclass that encapsulates all configuration
parameters for a Catan game board. It defines board geometry, resource distributions,
building costs, game rules, and visual presentation settings in a centralized,
immutable configuration object.

The BoardLayout class serves as the foundation for board generation and game setup,
providing default values that represent a standard Catan game while allowing
customization for variants and AI training scenarios.

Classes:
    BoardLayout: Dataclass containing all board configuration parameters.

Examples:
    Basic usage with default settings:
        >>> layout = BoardLayout()
        >>> print(f"Board has {layout.rings} rings with scale {layout.scale}")
        >>> print(f"Winning score: {layout.winning_score}")

    Creating custom board configuration:
        >>> custom_layout = BoardLayout(
        ...     rings=4,
        ...     winning_score=12,
        ...     street_cost='BGW'
        ... )
        >>> board_dict = custom_layout.asdict()

    Making modifications with copy:
        >>> base_layout = BoardLayout()
        >>> modified_layout = base_layout.copy()
        >>> # Modify the copy without affecting the original

Attributes:
    All attributes are defined within the BoardLayout dataclass.

Note:
    This module uses the dataclasses module for automatic generation of
    special methods like __init__, __repr__, and __eq__. The configuration
    is designed to be immutable after creation to ensure consistency.

Author:
    Rob Hendriks

Version:
    1.0.0
"""

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field, asdict

@dataclass
class BoardLayout:
    """Comprehensive configuration dataclass for Catan game board setup.
    
    This dataclass encapsulates all parameters needed to define a complete Catan
    game board, including physical layout, resource distribution, building costs,
    game rules, and visualization settings. It provides a centralized configuration
    that can be easily serialized, copied, and modified for different game variants.
    
    The class uses sensible defaults that represent a standard Catan game, making
    it easy to create boards for both regular gameplay and AI training scenarios.
    All string-based costs and requirements use single-character codes for
    efficient storage and processing.
    
    Attributes:
        tile_layout (str): Single-character string defining resource types for each
            tile in order. Characters: 'D'=Desert, 'S'=Sheep, 'W'=Wood, 'G'=Grain,
            'O'=Ore, 'B'=Brick. Length must match total tiles for the ring count.
        values (list[int]): Dice values (2-12) assigned to each tile. Index
            corresponds to tile_layout. Desert tiles typically have value 0.
        scale (int): Geometric scale factor for coordinate calculations. Affects
            visual rendering and precise positioning of board elements.
        rings (int): Number of concentric hexagonal rings around center tile.
            Determines board size: 1 ring = 7 tiles, 2 rings = 19 tiles, etc.
        resource_types (str): Single-character codes for all resource types in
            the game. Used for cost calculations and resource management.
        street_cost (str): Resources required to build a street. Each character
            represents one required resource card.
        village_cost (str): Resources required to build a village. Each character
            represents one required resource card.
        town_cost (str): Resources required to upgrade village to town. Each
            character represents one required resource card.
        development_card_cost (str): Resources required to buy a development card.
            Each character represents one required resource card.
        winning_score (int): Victory points needed to win the game. Standard
            Catan uses 10 points.
        dice_value_to_hand_in_cards (int): Dice roll value that triggers the
            robber and forces players to discard half their cards.
        max_available_villages (int): Maximum villages each player can build.
            Limited by physical game pieces.
        max_available_towns (int): Maximum towns each player can build.
            Limited by physical game pieces.
        max_available_streets (int): Maximum streets each player can build.
            Limited by physical game pieces.
        longest_street_minimum (int): Minimum connected street length required
            to claim the "Longest Road" bonus.
        plot_colors_players (list[str]): Color names for visual representation
            of each player's pieces on the board.
        plot_labels_for_resources (list[str]): Human-readable names for resource
            types, used in visualization and logging.
        plot_max_card_in_hand_per_type (int): Maximum resource cards of one type
            to display in hand visualizations.
        no_of_players (int): Number of players in the game. Affects setup and
            certain game mechanics.
    
    Examples:
        Creating a standard board:
            >>> layout = BoardLayout()
            >>> print(f"Standard board: {layout.rings} rings, {len(layout.values)} tiles")
            >>> print(f"Street cost: {layout.street_cost}")
            
        Creating a custom board for AI training:
            >>> ai_layout = BoardLayout(
            ...     tile_layout='SWGOBDSWGOBSWGO',  # Custom resource distribution
            ...     values=[0,6,8,10,9,5,4,11,3,2,12,6,8,4,11],
            ...     rings=2,  # Smaller board for faster training
            ...     winning_score=8  # Shorter games
            ... )
            
        Modifying an existing layout:
            >>> base = BoardLayout()
            >>> modified = base.copy()
            >>> # Now modify specific attributes as needed
            
        Converting to dictionary for serialization:
            >>> layout_dict = layout.asdict()
            >>> # Save to JSON, pass to other functions, etc.
    
    Note:
        This is a dataclass, so standard dataclass methods (__init__, __repr__,
        __eq__, etc.) are automatically generated. The configuration should be
        treated as immutable after creation - use copy() to create variants.
        
        Resource codes follow the convention: B=Brick, D=Desert, G=Grain,
        O=Ore, S=Sheep, W=Wood. All costs are represented as strings where
        each character indicates one required resource card.
        
        Type annotations have been corrected to match actual data types.
        Previously some integer attributes were incorrectly annotated as str.
    """
    tile_layout: str = 'DSWSWSWWGSOBGBGOBOG'  # Resource tiles: D=Desert, S=Sheep, W=Wood, G=Grain, O=Ore, B=Brick
    values: list = field(default_factory=lambda: [0,11,3,6,5,4, 9,10,8,4,11,12,9,10,8,3,6,2,5])  # Dice values for each tile (0 for desert)
    scale: int = 5  # Geometric scale factor for coordinate calculations
    rings: int = 3  # Number of hexagonal rings (1=7 tiles, 2=19 tiles, 3=37 tiles)
    resource_types: str = 'BDGOSW'  # All resource types: Brick, Desert, Grain, Ore, Sheep, Wood
    resource_type_names = ['Brick', 'Desert', 'Grain', 'Ore', 'Sheep', 'Wood']  # Human-readable resource names
    street_cost: str = 'BW'  # Street costs: 1 Brick + 1 Wood
    village_cost: str = 'BGSW'  # Village costs: 1 Brick + 1 Grain + 1 Sheep + 1 Wood
    town_cost: str = 'GGOOO'  # Town upgrade costs: 2 Grain + 3 Ore
    development_card_cost: str  = 'GOS'  # Dev card costs: 1 Grain + 1 Ore + 1 Sheep
    winning_score: int = 10  # Victory points needed to win
    dice_value_to_hand_in_cards: int = 7  # Dice value that triggers robber/discard
    max_available_villages: int = 5  # Maximum villages per player
    max_available_towns: int = 5  # Maximum towns per player
    max_available_streets: int = 12  # Maximum streets per player
    longest_street_minimum: int = 3  # Minimum connected streets for "Longest Road" bonus
    plot_colors_players = ['blue','green','red','yellow','purple','pink','orange']  # Player colors for visualization
    plot_labels_for_resources = ['Brick', 'Desert','Grain', 'Ore', 'Sheep', 'Wood']  # Human-readable resource names
    plot_max_card_in_hand_per_type: int = 7  # Max cards of one type to show in hand displays
    no_of_players: int = 4  # Number of players (affects setup and game mechanics)

    def asdict(self):
        """Convert the BoardLayout instance to a dictionary representation.
        
        Creates a dictionary containing all attributes and their values from
        this BoardLayout instance. This method is essential for serialization,
        configuration saving/loading, and passing board parameters to functions
        that expect dictionary inputs.
        
        The returned dictionary maintains the same structure as the dataclass
        fields, with attribute names as keys and their current values as
        dictionary values. This enables easy JSON serialization, configuration
        file generation, and parameter passing.
        
        Returns:
            dict: Dictionary representation where keys are attribute names
                (str) and values are the corresponding attribute values.
                All nested objects are also converted to dictionaries.
                
        Example:
            >>> layout = BoardLayout()
            >>> config_dict = layout.asdict()
            >>> print(config_dict['tile_layout'])
            'DSWSWSWWGSOBGBGOBOG'
            >>> print(config_dict['rings'])
            3
            >>> 
            >>> # Use for JSON serialization
            >>> import json
            >>> json_string = json.dumps(config_dict)
            >>> 
            >>> # Use for parameter passing
            >>> def create_board(config):
            ...     return BoardStructure(BoardLayout(**config))
            >>> board = create_board(config_dict)
            
        Note:
            This method uses the dataclasses.asdict() function internally,
            which recursively converts nested dataclass instances to
            dictionaries as well. The resulting dictionary is a deep copy
            of the current state.
        """
        return asdict(self)

    def copy(self):
        """Create a deep copy of the BoardLayout instance.
        
        Returns a new BoardLayout instance with identical configuration to
        the current instance. This is essential when you need to create
        variations of a board configuration without modifying the original.
        
        The copy operation creates a completely independent instance, so
        modifications to the copy will not affect the original BoardLayout
        instance. This is particularly useful for creating board variants,
        testing different configurations, or maintaining configuration history.
        
        Returns:
            BoardLayout: A new BoardLayout instance with identical attribute
                values to the current instance. All mutable attributes are
                deep-copied to ensure complete independence.
                
        Example:
            >>> base_layout = BoardLayout()
            >>> custom_layout = base_layout.copy()
            >>> 
            >>> # Modify the copy without affecting the original
            >>> custom_layout.winning_score = 12
            >>> custom_layout.rings = 4
            >>> 
            >>> # Original remains unchanged
            >>> assert base_layout.winning_score == 10
            >>> assert base_layout.rings == 3
            >>> 
            >>> # Use for creating variants
            >>> training_layouts = []
            >>> for score in [8, 10, 12]:
            ...     variant = base_layout.copy()
            ...     variant.winning_score = score
            ...     training_layouts.append(variant)
            
        Note:
            This method internally uses asdict() to serialize the current
            state and then creates a new instance from that dictionary,
            ensuring a proper deep copy of all attributes including
            mutable collections like lists.
        """
        return BoardLayout(**self.asdict())
  