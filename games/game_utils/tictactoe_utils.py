import numpy as np
import torch

class Board:
    def __init__(self, n=3):
        self.n = n
        self.pieces = torch.zeros((n, n), dtype=torch.int)

    def __getitem__(self, index):
        return self.pieces[index]

    def get_legal_moves(self):
        return torch.nonzero(self.pieces == 0).tolist()

    def has_legal_moves(self):
        return torch.any(self.pieces == 0)

    def is_win(self, color):
        n = self.pieces.shape[0]
        
        # Check rows and columns
        if torch.any(torch.all(self.pieces == color, dim=1)) or torch.any(torch.all(self.pieces == color, dim=0)):
            return True
        
        # Check diagonals
        if torch.all(torch.diag(self.pieces) == color) or torch.all(torch.diag(torch.fliplr(self.pieces)) == color):
            return True
        
        return False

    def execute_move(self, move, color):
        x, y = move
        self.pieces[x, y] = color


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanTicTacToePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i/self.game.n), int(i%self.game.n))
        while True: 
            # Python 3.x
            a = input()
            # Python 2.x 
            # a = raw_input()

            x,y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a