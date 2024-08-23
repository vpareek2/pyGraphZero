import numpy as np

class Board():
    def __init__(self, n=3):
        self.n = n
        self.pieces = [None] * self.n
        for i in range(self.n):
            self.pieces[i] = [0] * self.n

    def __getitem__(self, index):
        return self.pieces[index]

    def get_legal_moves(self):
        moves = []
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    moves.append((x, y))
        return moves

    def has_legal_moves(self):
        return len(self.get_legal_moves()) > 0

    def execute_move(self, move, player):
        x, y = move
        self[x][y] = player

    def is_win(self, player):
        # Check rows, columns and diagonals
        for i in range(self.n):
            if all(self[i][j] == player for j in range(self.n)):
                return True
            if all(self[j][i] == player for j in range(self.n)):
                return True
        if all(self[i][i] == player for i in range(self.n)):
            return True
        if all(self[i][self.n-1-i] == player for i in range(self.n)):
            return True
        return False

    def is_draw(self):
        return not self.has_legal_moves() and not self.is_win(1) and not self.is_win(-1)

    def get_board(self):
        return np.array(self.pieces)