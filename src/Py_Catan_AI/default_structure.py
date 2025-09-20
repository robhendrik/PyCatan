from Py_Catan_AI.board_layout import BoardLayout
from Py_Catan_AI.board_structure import BoardStructure
from Py_Catan_AI.catan_player import generate_default_players

default_board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')
default_structure = BoardStructure(board_layout=default_board_layout)
default_players = generate_default_players(structure=default_structure)