import torch
from games.game_utils.connect4_utils import Board

class Connect4Game:
    def __init__(self, height=6, width=7, win_length=4):
        self.height = height
        self.width = width
        self.win_length = win_length

    def get_init_board(self):
        return torch.zeros((self.height, self.width), dtype=torch.float32)

    def get_board_size(self):
        return (6, 7)  # (rows, columns)

    def get_action_size(self):
        return self.width

    def get_next_state(self, board, player, action):
        if board.dim() == 2:  # Single board
            b = Board(self.height, self.width, self.win_length)
            b.pieces = board.cpu().numpy()
            b.add_stone(action, player)
            return (torch.tensor(b.pieces, dtype=torch.float32, device=board.device), -player)
        elif board.dim() == 3:  # Batched boards
            next_boards = []
            for i in range(board.shape[0]):
                b = Board(self.height, self.width, self.win_length)
                b.pieces = board[i].cpu().numpy()
                b.add_stone(action[i], player)
                next_boards.append(torch.tensor(b.pieces, dtype=torch.float32, device=board.device))
            return (torch.stack(next_boards), -player)
        else:
            raise ValueError("Unsupported board dimension")

    def get_valid_moves(self, board, player):
        if board.dim() == 2:  # Single board
            b = Board(self.height, self.width, self.win_length)
            b.pieces = board.cpu().numpy()
            return torch.tensor(b.get_valid_moves(), dtype=torch.float32, device=board.device)
        elif board.dim() == 3:  # Batched boards
            valid_moves = []
            for single_board in board:
                b = Board(self.height, self.width, self.win_length)
                b.pieces = single_board.cpu().numpy()
                valid_moves.append(b.get_valid_moves())
            return torch.tensor(valid_moves, dtype=torch.float32, device=board.device)
        else:
            raise ValueError("Unsupported board dimension")

    def get_game_ended(self, board, player):
        if board.dim() == 3:  # Batch of boards
            results = []
            for single_board in board:
                b = Board(self.height, self.width, self.win_length)
                b.pieces = single_board.cpu().numpy()
                if b.is_win(player):
                    results.append(1)
                elif b.is_win(-player):
                    results.append(-1)
                elif b.has_legal_moves():
                    results.append(0)
                else:
                    results.append(1e-4)  # draw has a very little value
            return torch.tensor(results, dtype=torch.float32, device=board.device)
        else:  # Single board
            b = Board(self.height, self.width, self.win_length)
            b.pieces = board.cpu().numpy()
            if b.is_win(player):
                return torch.tensor(1, dtype=torch.float32, device=board.device)
            if b.is_win(-player):
                return torch.tensor(-1, dtype=torch.float32, device=board.device)
            if b.has_legal_moves():
                return torch.tensor(0, dtype=torch.float32, device=board.device)
            return torch.tensor(1e-4, dtype=torch.float32, device=board.device)  # draw has a very little value

    def get_canonical_form(self, board, player):
        if board.dim() == 2:  # Single board
            return player * board
        elif board.dim() == 3:  # Batched boards
            return player.view(-1, 1, 1) * board
        else:
            raise ValueError("Unsupported board dimension")

    def get_symmetries(self, board, pi):
        return [(board, pi), (torch.fliplr(board), torch.flipud(pi))]

    def string_representation(self, board):
        return board.cpu().numpy().tobytes()

    @staticmethod
    def display(board):
        board_np = board.cpu().numpy()
        print(" ", end="")
        for j in range(board_np.shape[1]):
            print(f" {j}", end="")
        print()

        for i in range(board_np.shape[0]):
            print(f"{i}|", end="")
            for j in range(board_np.shape[1]):
                if board_np[i][j] == 1:
                    print("X ", end="")
                elif board_np[i][j] == -1:
                    print("O ", end="")
                else:
                    print(". ", end="")
            print("|")

        print(" ", end="")
        for j in range(board_np.shape[1]):
            print("--", end="")
        print("-")
