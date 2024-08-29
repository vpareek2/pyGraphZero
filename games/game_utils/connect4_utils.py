import torch

class Board:
    def __init__(self, height, width, win_length):
        self.height = height
        self.width = width
        self.win_length = win_length
        self.pieces = torch.zeros((height, width), dtype=torch.float32)

    def add_stone(self, column, player):
        for row in range(self.height - 1, -1, -1):
            if self.pieces[row][column] == 0:
                self.pieces[row][column] = player
                return True
        return False

    def get_valid_moves(self):
        return (self.pieces[0] == 0).float()

    def has_legal_moves(self):
        return self.get_valid_moves().sum() > 0

    def is_win(self, player):
        # Check horizontal locations
        for row in range(self.height):
            for col in range(self.width - self.win_length + 1):
                if torch.all(self.pieces[row, col:col+self.win_length] == player):
                    return True

        # Check vertical locations
        for row in range(self.height - self.win_length + 1):
            for col in range(self.width):
                if torch.all(self.pieces[row:row+self.win_length, col] == player):
                    return True

        # Check positively sloped diagonals
        for row in range(self.height - self.win_length + 1):
            for col in range(self.width - self.win_length + 1):
                if torch.all(torch.diagonal(self.pieces[row:row+self.win_length, col:col+self.win_length]) == player):
                    return True

        # Check negatively sloped diagonals
        for row in range(self.win_length - 1, self.height):
            for col in range(self.width - self.win_length + 1):
                if torch.all(torch.diagonal(torch.fliplr(self.pieces[row-self.win_length+1:row+1, col:col+self.win_length])) == player):
                    return True

        return False
