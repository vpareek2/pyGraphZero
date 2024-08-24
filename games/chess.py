import chess
import numpy as np

class ChessGame:
    def __init__(self):
        self.board = chess.Board()

    def get_init_board(self):
        self.board.reset()
        return self.board_to_numpy()

    def get_board_size(self):
        return (8, 8)  # Standard chess board size

    def get_action_size(self):
        return 8 * 8 * 73  # All possible moves: 64 squares * 73 possible moves from any square

    def get_next_state(self, board, player, action):
        """
        Input:
            board: current board (numpy array)
            player: current player (1 for white, -1 for black)
            action: action taken by current player

        Returns:
            next_board: board after applying action
            next_player: player who plays in the next turn
        """
        self.board = self.numpy_to_board(board)
        move = self.action_to_move(action)
        self.board.push(move)
        return self.board_to_numpy(), -player

    def get_valid_moves(self, board, player):
        self.board = self.numpy_to_board(board)
        valid_moves = [0] * self.get_action_size()
        for move in self.board.legal_moves:
            valid_moves[self.move_to_action(move)] = 1
        return np.array(valid_moves)

    def get_game_ended(self, board, player):
        self.board = self.numpy_to_board(board)
        if self.board.is_game_over():
            if self.board.is_checkmate():
                return 1 if self.board.turn == chess.BLACK else -1
            else:  # draw
                return 1e-4
        return 0

    def get_canonical_form(self, board, player):
        # In chess, the board is always from the perspective of the current player
        return board if player == 1 else np.flip(board, axis=0)

    def get_symmetries(self, board, pi):
        # Chess doesn't have symmetries like Connect4
        return [(board, pi)]

    def string_representation(self, board):
        self.board = self.numpy_to_board(board)
        return self.board.fen()

    @staticmethod
    def display(board):
        print(chess.Board(ChessGame.numpy_to_board(board).fen()))

    def board_to_numpy(self):
        # Convert chess.Board to a numpy array
        board_matrix = np.zeros((8, 8, 12), dtype=np.int8)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                color = int(piece.color)
                piece_type = piece.piece_type - 1
                board_matrix[i // 8, i % 8, piece_type + 6 * color] = 1
        return board_matrix

    @staticmethod
    def numpy_to_board(board_matrix):
        # Convert numpy array to chess.Board
        board = chess.Board(None)
        for i in range(8):
            for j in range(8):
                piece = np.argmax(board_matrix[i, j])
                if piece != 0:
                    color = chess.WHITE if piece > 5 else chess.BLACK
                    piece_type = (piece % 6) + 1
                    board.set_piece_at(8 * i + j, chess.Piece(piece_type, color))
        return board

    def action_to_move(self, action):
        # Convert action index to chess move
        from_square = action // 73
        to_square = (action % 73) // 8
        promotion = (action % 73) % 8
        return chess.Move(from_square, to_square, promotion)

    def move_to_action(self, move):
        # Convert chess move to action index
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion if move.promotion else 0
        return 73 * from_square + 8 * to_square + promotion

# Example usage
if __name__ == "__main__":
    game = ChessGame()
    board = game.get_init_board()
    player = 1
    
    print("Initial board:")
    game.display(board)
    
    # Make some moves
    for _ in range(5):
        valid_moves = game.get_valid_moves(board, player)
        action = np.random.choice(len(valid_moves), p=valid_moves/sum(valid_moves))
        board, player = game.get_next_state(board, player, action)
        print(f"\nAfter move:")
        game.display(board)
    
    print("\nGame ended:", game.get_game_ended(board, player))