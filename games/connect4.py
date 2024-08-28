# IMPORTANT UPDATE TO USE TENSORS

import numpy as np
from games.game_utils.connect4_utils import Board

class Connect4Game:
    def __init__(self, height=6, width=7, win_length=4):
        self.height = height
        self.width = width
        self.win_length = win_length

    def get_init_board(self):
        return np.zeros((self.height, self.width), dtype=np.int32)

    def get_board_size(self):
        return (self.height, self.width)

    def get_action_size(self):
        return self.width

    def get_next_state(self, board, player, action):
        b = Board(self.height, self.width, self.win_length)
        b.pieces = np.copy(board)
        b.add_stone(action, player)
        return (b.pieces, -player)

    def get_valid_moves(self, board, player):
        b = Board(self.height, self.width, self.win_length)
        b.pieces = np.copy(board)
        return b.get_valid_moves()

    def get_game_ended(self, board, player):
        b = Board(self.height, self.width, self.win_length)
        b.pieces = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        return 1e-4  # draw has a very little value

    def get_canonical_form(self, board, player):
        return player * board

    def get_symmetries(self, board, pi):
        return [(board, pi), (np.fliplr(board), np.flipud(pi))]

    def string_representation(self, board):
        return board.tobytes()

    @staticmethod
    def display(board):
        print(" ", end="")
        for j in range(board.shape[1]):
            print(f" {j}", end="")
        print()

        for i in range(board.shape[0]):
            print(f"{i}|", end="")
            for j in range(board.shape[1]):
                if board[i][j] == 1:
                    print("X ", end="")
                elif board[i][j] == -1:
                    print("O ", end="")
                else:
                    print(". ", end="")
            print("|")

        print(" ", end="")
        for j in range(board.shape[1]):
            print("--", end="")
        print("-")
