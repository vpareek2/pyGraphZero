import torch
import chess

class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_init_board(self):
        return self.board_to_tensor()

    def get_board_size(self):
        return (8, 8)

    def get_action_size(self):
        return 8 * 8 * 73  # 64 squares, 73 possible moves per square (including promotions)

    def action_to_move(self, action):
        from_square = action // 73
        to_square = (action % 73) % 64
        promotion = (action % 73) // 64 + 1 if action % 73 >= 64 else None
        
        # Debug print statements
        print(f"Action: {action}")
        print(f"From square: {from_square}")
        print(f"To square: {to_square}")
        print(f"Promotion: {promotion}")
        
        # Check if squares are valid
        if from_square < 0 or from_square > 63 or to_square < 0 or to_square > 63:
            print("Invalid squares detected")
            return None
        
        return chess.Move(from_square, to_square, promotion)

    def get_valid_moves(self, board, player):
        chess_board = self.tensor_to_board(board)
        valid_moves = torch.zeros(self.get_action_size(), dtype=torch.int8, device=self.device)
        for move in chess_board.legal_moves:
            valid_moves[self.move_to_action(move)] = 1
        return valid_moves

    def get_game_ended(self, board, player):
        chess_board = self.tensor_to_board(board)
        if chess_board.is_game_over():
            if chess_board.is_checkmate():
                return torch.tensor(-player, dtype=torch.float32, device=self.device)
            elif chess_board.is_stalemate() or chess_board.is_insufficient_material() or chess_board.is_seventyfive_moves() or chess_board.is_fivefold_repetition():
                return torch.tensor(1e-4, dtype=torch.float32, device=self.device)  # draw has a very little value
        return torch.tensor(0, dtype=torch.float32, device=self.device)

    def get_canonical_form(self, board, player):
        return board * player

    def get_symmetries(self, board, pi):
        # Chess has no symmetries
        return [(board, pi)]

    def string_representation(self, board):
        return self.tensor_to_board(board).fen()

    def board_to_tensor(self, board=None):
        if board is None:
            board = self.board
        tensor_board = torch.zeros((8, 8, 12), dtype=torch.int8, device=self.device)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                color = int(piece.color)
                piece_type = piece.piece_type - 1
                tensor_board[square // 8, square % 8, piece_type + 6 * color] = 1
        return tensor_board

    def tensor_to_board(self, tensor_board):
        board = chess.Board(None)
        for square in chess.SQUARES:
            piece_index = torch.argmax(tensor_board[square // 8, square % 8]).item()
            if piece_index < 12:
                color = chess.WHITE if piece_index < 6 else chess.BLACK
                piece_type = (piece_index % 6) + 1
                board.set_piece_at(square, chess.Piece(piece_type, color))
        return board

    def move_to_action(self, move):
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion
        action = from_square * 73 + to_square
        if promotion:
            action += (promotion - 1) * 64 + 64
        return action

    def action_to_move(self, action):
        from_square = action // 73
        to_square = (action % 73) % 64
        promotion = (action % 73) // 64 + 1 if action % 73 >= 64 else None
        
        # Debug print statements
        print(f"Action: {action}")
        print(f"From square: {from_square}")
        print(f"To square: {to_square}")
        print(f"Promotion: {promotion}")
        
        # Check if squares are valid
        if from_square < 0 or from_square > 63 or to_square < 0 or to_square > 63:
            print("Invalid squares detected")
            return None
        
        return chess.Move(from_square, to_square, promotion)

    @staticmethod
    def display(board):
        print(chess.Board(ChessGame().string_representation(board)))