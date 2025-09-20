import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.animation as animation
import matplotlib.image as mpimg
import numpy as np
import textwrap
from dataclasses import dataclass
from importlib.resources import files

plt.rcParams['animation.ffmpeg_path'] = r"C:\ffmpeg\ffmpeg-8.0-full_build\bin\ffmpeg.exe" 
FIGSIZE = (15, 10)
DEFAULT_PATH_TO_DOCS = 'Py_Catan_AI.docs'
# === Layout ===
board_position = [0.15, 0.15, 0.4, 0.7]
bar_positions = [[0.05, 0.75, 0.1, 0.2],
                 [0.55, 0.75, 0.1, 0.2],
                 [0.05, 0.15, 0.1, 0.2],
                 [0.55, 0.15, 0.1, 0.2]]
text_box = [0.65, 0.05, 0.45, 0.9]

class ChatWindow:
    def __init__(self, fig, game, rect=text_box, max_messages=10):
        self.ax = fig.add_axes(rect)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, max_messages)
        self.ax.axis("off")
        self.max_messages = max_messages
        self.messages = []
        self.colors = game.structure.plot_colors_players

    def add_comment(self, player_id, player_name, text):
        self.messages.append((player_id, player_name, text))
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, self.max_messages)
        self.ax.axis("off")
        top_y = self.max_messages - 0.2
        for i, (pid, name, msg) in enumerate(reversed(self.messages)):
            wrapped = textwrap.wrap(f"{name}: {msg}", width=60)
            bubble_height = 0.3 + 0.3 * len(wrapped)
            y_bottom = top_y - bubble_height
            if y_bottom < 0:
                break
            bubble = FancyBboxPatch((0.05, y_bottom), 0.9, bubble_height,
                                    boxstyle="round,pad=0.15",
                                    facecolor=self.colors[pid % len(self.colors)],
                                    edgecolor="black")
            self.ax.add_patch(bubble)
            weight = "bold" if i == 0 else "normal"
            for j, line in enumerate(wrapped):
                self.ax.text(0.07, y_bottom + bubble_height - 0.25 - j * 0.3,
                             line, va="top", ha="left", fontsize=8, fontweight=weight)
            top_y = y_bottom - 0.1
        self.ax.figure.canvas.draw_idle()

# ==== helper functions ====

def _plot_board_indicators_2(structure, dice, ax):
    for tile_number, tile in enumerate(structure._tile_coordinates):
        text = structure.tile_layout[tile_number] + '  /  ' + str(structure.values[tile_number])
        if structure.values[tile_number] == dice and dice != 0:
            ax.plot([tile[0]],[tile[1]],marker='H', color='lightgrey', markersize=80)
            ax.plot([tile[0]],[tile[1]],marker=r"$ {} $".format(text), color='red', markersize=25)
        else:
            ax.plot([tile[0]],[tile[1]],marker='H', color='lightgrey', markersize=60)
            ax.plot([tile[0]],[tile[1]],marker=r"$ {} $".format(text), color='black', markersize=15)
    for node_number,node in enumerate(structure._node_coordinates):   
        ax.plot([node[0]],[node[1]],marker='o', color='darkgrey', markersize=10)                       
        ax.plot([node[0]],[node[1]],marker=r"$ {} $".format(node_number), color='grey', markersize=4)
    for edge_number,edge in enumerate(structure._edge_coordinates):
        middle = (edge[0]+edge[1])/2
        ax.plot([middle[0]],[middle[1]],marker='s', color='grey', markersize=10)            
        ax.plot([middle[0]],[middle[1]],marker=r"$ {} $".format(edge_number), color='darkgrey', markersize=4)
    return

def _add_players_to_board(structure, ax, vector):
    colors = structure.plot_colors_players
    @dataclass
    class Place:
        streets: list
        villages: list
        towns: list
        hand: list
        name: str
    players = []
    for i in range(4):
        p = Place(
            streets=vector[structure.vector_indices["edges"]] == i + 1,
            villages=vector[structure.vector_indices["nodes"]] == i + 1,
            towns=vector[structure.vector_indices["nodes"]] == i + 5,
            hand=vector[structure.vector_indices["hand_for_player"][i]],
            name=f"Player {i+1}",
        )
        players.append(p)
    for p, color in zip(players, colors):
        for node_number,node in enumerate(structure._node_coordinates):   
            if p.towns[node_number]:
                ax.plot([node[0]],[node[1]],marker='*', color=color, markersize=20,zorder=100)      
            if p.villages[node_number]:
                ax.plot([node[0]],[node[1]],marker='o', color=color, markersize=15,zorder=100)       
        for edge_number,edge in enumerate(structure._edge_coordinates):
            if p.streets[edge_number]:
                start = edge[0]- 0.25 *(edge[0]-edge[1])
                end = edge[0]- 0.75 *(edge[0]-edge[1])
                ax.plot([start[0],end[0]],[start[1],end[1]], color=color, linewidth=6.0,zorder=100) 

def plot_board_positions_with_indices_from_vector_2(structure, input_vector, names, active_player=0, info=0, fig=None):
    new_vector = input_vector.copy()
    if fig is None:
        fig = plt.figure(figsize=FIGSIZE)
    if len(fig.axes) == 0:
        ax_board = fig.add_axes(board_position)
    else:
        ax_board = fig.axes[0]
        ax_board.cla()
        ax_board.set_position(board_position)
    ax_board.axis('off')
    _plot_board_indicators_2(structure, info['dice result'], ax=ax_board)
    _add_players_to_board(structure, ax_board, new_vector)
    ax_board.set_aspect("equal")
    for spine in ax_board.spines.values():
        spine.set_visible(False)
    ax_board.set_xticks([])
    ax_board.set_yticks([])
    if info['stage']['phase'] == 'initial_placement':
        ax_board.set_title("Initial Placement Phase: " + " by " + names[info['stage']['active_player']], fontsize=12)
    else:
        ax_board.set_title("Game Play Phase: " + names[info['stage']['active_player']].strip() + "'s turn. Round: "+ str(info['rounds']+1) + " Action: " + str(info['action in round']), fontsize=12)
    hand_labels = structure.plot_labels_for_resources
    hands = [new_vector[structure.vector_indices["hand_for_player"][i]] for i in range(4)]
    for idx, (hand, color, name) in enumerate(zip(hands, structure.plot_colors_players, names)):
        if idx >= 4: break
        if len(fig.axes) <= idx + 1:
            bar_ax = fig.add_axes(bar_positions[idx])
        else:
            bar_ax = fig.axes[idx + 1]
            bar_ax.cla()
            bar_ax.set_position(bar_positions[idx])
        bar_ax.bar(hand_labels, structure.plot_max_card_in_hand_per_type, color="lightgrey", alpha=0.3)
        bar_ax.bar(hand_labels, hand, color=color)
        bar_ax.set_title(name, fontsize=8)
        bar_ax.set_xticks(np.arange(len(hand_labels)))
        bar_ax.set_xticklabels(hand_labels, rotation=90, fontsize=7)
        bar_ax.set_yticks([])
        bar_ax.set_ylim(0, structure.plot_max_card_in_hand_per_type)
        for spine in bar_ax.spines.values():
            spine.set_visible(False)
        bar_ax.hlines( y=list(range(1,structure.plot_max_card_in_hand_per_type)),
            xmin=-1,xmax=len(hand_labels),
            colors=['w'] *structure.plot_max_card_in_hand_per_type,
            linestyles=['-']*structure.plot_max_card_in_hand_per_type)
        bar_ax.text(-0.5, -3.75, 'Score: '+str(info['score'][idx]), ha='left', va='bottom')
        bar_ax.text(-0.5, -4.5, 'Street Length: '+str(info['street_length'][idx]), ha='left', va='bottom')

    return fig



def plot_from_log(structure, names, game_log, game):
    plt.ion()
    fig, chat = None, None
    for entry in game_log:
        vector, info, message = entry['vector'], entry['info'], entry['message']
        fig = plot_board_positions_with_indices_from_vector_2(
            structure, vector, names, info['stage']['active_player'], info, fig
        )
        if chat is None:
            chat = ChatWindow(fig, game, rect=text_box, max_messages=10)
        chat.add_comment(info['stage']['active_player'], names[info['stage']['active_player']], message)
        plt.pause(0.01)
    plt.ioff()
    plt.show()

def video_from_log(game_log: any,filename="game_progress.mp4"):
    structure = game_log.structure
    game = game_log.game
    names = [p.name for p in game_log.players]
    # ==== THIS IS UGLY, SAME NAME FOR DATACLASS AS FOR DATAFRAME ====
    game_log = game_log.log

    fig = plt.figure(figsize=FIGSIZE)
    chat = None
    path_to_intro_img = files(DEFAULT_PATH_TO_DOCS).joinpath('ChatGPT Image 13 sep 2025, 06_34_30.png')
    intro_img = mpimg.imread(path_to_intro_img)

    def update(i):
        nonlocal chat

        if i == 0:
            # Clean intro frame
            fig.clf()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(intro_img)
            ax.axis("off")
            ax.text(
                0.5, 0.9, "Py_Catan, the AI version",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=24, color="white", weight="bold"
            )
            return fig.axes

        # ---- regular frames ----

        entry = game_log.iloc[i - 1].to_dict()
        # ===== THIS IS NOT EFFICIENT, SHOULD NOT RECREATE INFO HERE =====
        # entry is in 'game order' so teh scores, street length, street/village/town indices in the vector
        # correspond to the order in 'names'
        info_for_plotting = {
            'stage': {'active_player': entry['active_player'], 'phase': entry['stage']},
            'rounds': entry['rounds'],
            'action in round': entry['action_in_round'],
            'dice result': entry['dice_result'],
            'terminated': entry['terminated'],
            'truncated': entry['truncated'],
            'street_length': np.array(entry['street_length']),
            'score': np.array(entry['score']),
        }

        # 1) Draw/refresh board FIRST so it claims fig.axes[0]
        plot_board_positions_with_indices_from_vector_2(
            structure, entry['vector'], names, info_for_plotting['stage']['active_player'], info_for_plotting, fig
        )

        # 2) Now create/update chat so it isn't axes[0] (won't be cleared by the board)
        if chat is None:
            chat = ChatWindow(fig, game, rect=text_box, max_messages=10)
            # Optional: make sure chat sits on top of others
            chat.ax.set_zorder(10)

        ap = int(entry['active_player'])
        active_name = entry['player_names'][entry['active_player']]
        if type(entry['comments']) is dict and active_name in entry['comments']:
            active_comment = entry['comments'][active_name]
            chat.add_comment(ap, active_name, str(active_comment))
        else:
            chat.add_comment(ap, active_name, str(entry['message']))

        return fig.axes

    ani = animation.FuncAnimation(fig, update, frames=len(game_log) + 1, interval=800, repeat=False)
    ani.save(filename, writer="ffmpeg", dpi=150)
    print(f"🎥 Video saved to {filename}")
    plt.close(fig)
    return

