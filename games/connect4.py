import numpy as np
from games.game_utils.connect4_utils import Board, DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_WIN_LENGTH

class Connect4Game:
    def __init__(self, height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH, win_length=DEFAULT_WIN_LENGTH):
        self.height = height
        self.width = width
        self.win_length = win_length
        self._base_board = Board(height, width, win_length)

    def get_init_board(self):
        return self._base_board.np_pieces

    def get_board_size(self):
        return (self.height, self.width)

    def get_action_size(self):
        return self.width

    def get_next_state(self, board, player, action):
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = Board(self.height, self.width, self.win_length, np_pieces=np.copy(board))
        b.add_stone(action, player)
        return b.np_pieces, -player

    def get_valid_moves(self, board, player):
        "Any zero value in top row in a valid move"
        return Board(self.height, self.width, self.win_length, np_pieces=board).get_valid_moves()

    def get_game_ended(self, board, player):
        print(f"Current board state:\n{board}")  # Debug print
        b = Board(self.height, self.width, self.win_length, np_pieces=board)
        winstate = b.get_win_state()
        print(f"Winstate: {winstate}")  # Debug print
        if winstate.is_ended:
            if winstate.winner is None:
                print("Draw detected")  # Debug print
                return 1e-4
            elif winstate.winner == player:
                print(f"Player {player} won")  # Debug print
                return +1
            elif winstate.winner == -player:
                print(f"Player {-player} won")  # Debug print
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', winstate)
        else:
            print("Game not ended")  # Debug print
            return 0

    def get_canonical_form(self, board, player):
        # Flip player from 1 to -1
        return board * player

    def get_symmetries(self, board, pi):
        """Board is left/right board symmetric"""
        return [(board, pi), (board[:, ::-1], pi[::-1])]

    def string_representation(self, board):
        return board.tobytes()

    @staticmethod
    def display(board):
        print(" -----------------------")
        print(' '.join(map(str, range(len(board[0])))))
        print(board)
        print(" -----------------------")