import torch
from games.game_utils.tictactoe_utils import Board

class TicTacToeGame:
    def __init__(self, n=3):
        self.n = n

    def get_init_board(self):
        b = Board(self.n)
        return torch.tensor(b.pieces, dtype=torch.float32)

    def get_board_size(self):
        return (self.n, self.n)

    def get_action_size(self):
        return self.n * self.n

    def get_next_state(self, board, player, action):
        if board.dim() == 2:  # Single board
            if action == self.n * self.n:
                return (board, -player)
            b = board.clone()
            move = (int(action / self.n), int(action % self.n))
            b[move] = player
            return (b, -player)
        elif board.dim() == 3:  # Batched boards
            next_boards = board.clone()
            for i in range(board.shape[0]):
                if action[i] != self.n * self.n:
                    move = (int(action[i] / self.n), int(action[i] % self.n))
                    next_boards[i, move[0], move[1]] = player
            return (next_boards, -player)
        else:
            raise ValueError("Unsupported board dimension")

    def get_valid_moves(self, board, player):
        if board.dim() == 2:  # Single board
            valids = torch.zeros(self.n * self.n + 1, dtype=torch.float32, device=board.device)
            empty_spots = (board == 0).flatten()
            valids[:self.n * self.n] = empty_spots.float()
            valids[-1] = 1  # Add pass move
            return valids
        elif board.dim() == 3:  # Batched boards
            empty_spots = (board == 0).view(board.shape[0], -1)
            pass_move = torch.ones((board.shape[0], 1), dtype=torch.float32, device=board.device)
            return torch.cat([empty_spots.float(), pass_move], dim=1)
        else:
            raise ValueError("Unsupported board dimension")

    def get_game_ended(self, board, player):
        def check_win(b):
            # Check rows, columns and diagonals
            for i in range(3):
                if torch.all(b[i] == player) or torch.all(b[:, i] == player):
                    return True
            if torch.all(torch.diag(b) == player) or torch.all(torch.diag(b.flip(1)) == player):
                return True
            return False

        if board.dim() == 3:  # Batch of boards
            results = []
            for single_board in board:
                if check_win(single_board):
                    results.append(1)
                elif check_win(-single_board):
                    results.append(-1)
                elif torch.any(single_board == 0):
                    results.append(0)
                else:
                    results.append(1e-4)  # draw has a very little value
            return torch.tensor(results, dtype=torch.float32, device=board.device)
        else:  # Single board
            if check_win(board):
                return torch.tensor(1, dtype=torch.float32, device=board.device)
            if check_win(-board):
                return torch.tensor(-1, dtype=torch.float32, device=board.device)
            if torch.any(board == 0):
                return torch.tensor(0, dtype=torch.float32, device=board.device)
            return torch.tensor(1e-4, dtype=torch.float32, device=board.device)  # draw has a very little value

    def get_canonical_form(self, board, player):
        if board.dim() == 2:  # Single board
            return player * board
        elif board.dim() == 3:  # Batched boards
            return player * board
        else:
            raise ValueError("Unsupported board dimension")

    def get_symmetries(self, board, pi):
        assert(len(pi) == self.n**2 + 1)  # 1 for pass
        pi_board = torch.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = torch.rot90(board, i)
                newPi = torch.rot90(pi_board, i)
                if j:
                    newB = torch.fliplr(newB)
                    newPi = torch.fliplr(newPi)
                l += [(newB, torch.cat([newPi.flatten(), pi[-1].unsqueeze(0)]))]
        return l

    def string_representation(self, board):
        return board.cpu().numpy().tobytes()

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
                piece = board[y][x].item()
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