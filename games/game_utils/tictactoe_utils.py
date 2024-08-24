import numpy as np

class Board:
    def __init__(self, n=3):
        self.n = n
        self.pieces = [[0 for _ in range(n)] for _ in range(n)]

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
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    return True
        return False

    def is_win(self, color):
        # Check rows and columns
        for i in range(self.n):
            if all(self[i][j] == color for j in range(self.n)) or \
               all(self[j][i] == color for j in range(self.n)):
                return True

        # Check diagonals
        if all(self[i][i] == color for i in range(self.n)) or \
           all(self[i][self.n-i-1] == color for i in range(self.n)):
            return True

        return False

    def execute_move(self, move, color):
        x, y = move
        assert self[x][y] == 0
        self[x][y] = color


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