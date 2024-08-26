import numpy as np
import torch
from games.game_utils.tictactoe_utils import Board

class TicTacToeGame:
    def __init__(self, n=3):
        self.n = n

    def get_init_board(self):
        b = Board(self.n)
        return np.array(b.pieces)

    def get_board_size(self):
        return (self.n, self.n)

    def get_action_size(self):
        return self.n * self.n + 1

    def get_next_state(self, board, player, action):
        if action == self.n * self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def get_valid_moves(self, board, player):
        if isinstance(board, np.ndarray):
            # Single board
            b = Board(board)
            legal_moves = b.get_legal_moves()
            valids = [1 if (i, j) in legal_moves else 0 for i in range(3) for j in range(3)]
            return np.array(valids)
        elif isinstance(board, torch.Tensor):
            # Batched boards
            empty_spots = (board == 0)
            legal_moves = empty_spots.view(board.shape[0], -1)
            return legal_moves
        else:
            raise ValueError("Unsupported board type")

    def get_game_ended(self, board, player):
        b = Board(self.n)
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
        assert(len(pi) == self.n**2 + 1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def string_representation(self, board):
        return board.tobytes()

    @staticmethod
    def display(board):
        n = board.shape[0]

        print("   ", end="")
        for y in range(n):
            print(y, "", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|", end="")
            for x in range(n):
                piece = board[y][x]
                if piece == -1:
                    print("X ", end="")
                elif piece == 1:
                    print("O ", end="")
                else:
                    if x == n:
                        print("-", end="")
                    else:
                        print("- ", end="")
            print("|")

        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")