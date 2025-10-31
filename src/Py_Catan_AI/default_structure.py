"""Default board layout, structure, and player set for quick starts.

This module exposes three ready-to-use objects intended for demos,
interactive exploration, and unit tests:

- ``default_board_layout``: a `BoardLayout` instance pre-populated with a
	standard tile arrangement suitable for quick experiments.
- ``default_structure``: a `BoardStructure` computed from
	``default_board_layout``; contains coordinates, indexing, and action
	space metadata used across the codebase.
- ``default_players``: a small list of player instances created by
	`generate_default_players` matching the structure's player count. Use
	these for running quick games, visualizations, or test harnesses.

Usage example::

		from Py_Catan_AI.default_structure import default_structure, default_players
		# Use default_structure to build a board or vectorize a state

Notes:
	- These objects are created at import time; creating many of them in a
		tight loop may be inefficient. For repeated experiments, prefer
		constructing your own `BoardLayout`/`BoardStructure` instances.
	- The defaults are intended as convenient examples, not canonical game
		configurations — feel free to replace `tile_layout` or other settings
		for different scenarios.
"""

from Py_Catan_AI.board_layout import BoardLayout
from Py_Catan_AI.board_structure import BoardStructure
from Py_Catan_AI.player import generate_default_players

# Default board layout chosen for demos and tests. Adjust `tile_layout`
# if you need a different resource distribution.
default_board_layout = BoardLayout(tile_layout='DSWOSBWWGSGSBGWOGOB')

# Precomputed board structure (coordinates, headers, action indices, etc.)
default_structure = BoardStructure(board_layout=default_board_layout)

# Default players (list) matching `default_structure.no_of_players`.
default_players = generate_default_players(structure=default_structure)