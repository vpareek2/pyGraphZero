import numpy as np

class Board:
    def __init__(self, height, width, win_length):
        self.height = height
        self.width = width
        self.win_length = win_length
        self.pieces = np.zeros((height, width), dtype=np.int32)

    def add_stone(self, column, player):
        for row in range(self.height - 1, -1, -1):
            if self.pieces[row][column] == 0:
                self.pieces[row][column] = player
                return True
        return False

    def get_valid_moves(self):
        return [1 if self.pieces[0][col] == 0 else 0 for col in range(self.width)]

    def has_legal_moves(self):
        return any(self.get_valid_moves())

    def is_win(self, player):
        # Check horizontal locations
        for row in range(self.height):
            for col in range(self.width - self.win_length + 1):
                if all(self.pieces[row][col + i] == player for i in range(self.win_length)):
                    return True

        # Check vertical locations
        for row in range(self.height - self.win_length + 1):
            for col in range(self.width):
                if all(self.pieces[row + i][col] == player for i in range(self.win_length)):
                    return True

        # Check positively sloped diagonals
        for row in range(self.height - self.win_length + 1):
            for col in range(self.width - self.win_length + 1):
                if all(self.pieces[row + i][col + i] == player for i in range(self.win_length)):
                    return True

        # Check negatively sloped diagonals
        for row in range(self.win_length - 1, self.height):
            for col in range(self.width - self.win_length + 1):
                if all(self.pieces[row - i][col + i] == player for i in range(self.win_length)):
                    return True

        return False
