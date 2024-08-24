from collections import namedtuple
import numpy as np

DEFAULT_HEIGHT = 6
DEFAULT_WIDTH = 7
DEFAULT_WIN_LENGTH = 4

WinState = namedtuple('WinState', 'is_ended winner')

class Board:
    def __init__(self, height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH, win_length=DEFAULT_WIN_LENGTH, np_pieces=None):
        "Set up initial board configuration."
        self.height = height
        self.width = width
        self.win_length = win_length

        if np_pieces is None:
            self.np_pieces = np.zeros([self.height, self.width], dtype=np.int32)
        else:
            self.np_pieces = np_pieces
            assert self.np_pieces.shape == (self.height, self.width)

    def add_stone(self, column, player):
        "Create copy of board containing new stone."
        available_idx, = np.where(self.np_pieces[:, column] == 0)
        if len(available_idx) == 0:
            raise ValueError(f"Can't play column {column} on board {self}")

        self.np_pieces[available_idx[-1]][column] = player

    def get_valid_moves(self):
        "Any zero value in top row in a valid move"
        return self.np_pieces[0] == 0

    def get_win_state(self):
        print(f"Board state in get_win_state:\n{self.np_pieces}")  # Debug print
        # Check for wins
        for player in [-1, 1]:
            player_pieces = self.np_pieces == player
            if (self._is_straight_winner(player_pieces) or
                self._is_straight_winner(player_pieces.transpose()) or
                self._is_diagonal_winner(player_pieces)):
                print(f"Win detected for player {player}")  # Debug print
                return WinState(True, player)

        # Check for draw
        if not self.get_valid_moves().any():
            print("Draw detected in get_win_state")  # Debug print
            return WinState(True, None)

        # Game is not ended yet.
        print("Game not ended in get_win_state")  # Debug print
        return WinState(False, None)

    def _is_straight_winner(self, player_pieces):
        """Checks if player_pieces contains a vertical or horizontal win."""
        run_lengths = [player_pieces[:, i:i + self.win_length].sum(axis=1)
                       for i in range(len(player_pieces) - self.win_length + 2)]
        max_run = max([x.max() for x in run_lengths])
        print(f"Max run length: {max_run}")  # Debug print
        return max_run >= self.win_length

    def _is_diagonal_winner(self, player_pieces):
        """Checks if player_pieces contains a diagonal win."""
        win_length = self.win_length
        for i in range(len(player_pieces) - win_length + 1):
            for j in range(len(player_pieces[0]) - win_length + 1):
                if all(player_pieces[i + x][j + x] for x in range(win_length)):
                    return True
            for j in range(win_length - 1, len(player_pieces[0])):
                if all(player_pieces[i + x][j - x] for x in range(win_length)):
                    return True
        return False

    def __str__(self):
        return str(self.np_pieces)
