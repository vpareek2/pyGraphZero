import unittest
import numpy as np
import chess
from games.chess import ChessGame

class TestChessGame(unittest.TestCase):
    def setUp(self):
        self.game = ChessGame()

    def test_init_board(self):
        init_board = self.game.get_init_board()
        self.assertEqual(init_board.shape, (8, 8, 12))
        self.assertEqual(np.sum(init_board), 32)  # 32 pieces on the board initially

    def test_board_size(self):
        self.assertEqual(self.game.get_board_size(), (8, 8))

    def test_action_size(self):
        self.assertEqual(self.game.get_action_size(), 8 * 8 * 73)

    def test_get_next_state(self):
        init_board = self.game.get_init_board()
        next_board, next_player = self.game.get_next_state(init_board, 1, 4672)  # e2e4
        self.assertEqual(np.sum(next_board), 32)  # Still 32 pieces
        self.assertEqual(next_player, -1)  # Player changed

    def test_get_valid_moves(self):
        init_board = self.game.get_init_board()
        valid_moves = self.game.get_valid_moves(init_board, 1)
        self.assertEqual(np.sum(valid_moves), 20)  # 20 valid moves in initial position

    def test_get_game_ended(self):
        init_board = self.game.get_init_board()
        self.assertEqual(self.game.get_game_ended(init_board, 1), 0)  # Game not ended

    def test_get_canonical_form(self):
        init_board = self.game.get_init_board()
        canonical_board = self.game.get_canonical_form(init_board, -1)
        self.assertTrue(np.array_equal(canonical_board, -init_board))

    def test_get_symmetries(self):
        init_board = self.game.get_init_board()
        symmetries = self.game.get_symmetries(init_board, [])
        self.assertEqual(len(symmetries), 1)  # Chess has no symmetries

    def test_string_representation(self):
        init_board = self.game.get_init_board()
        fen = self.game.string_representation(init_board)
        self.assertEqual(fen, chess.STARTING_FEN)

    def test_board_conversions(self):
        init_board = self.game.get_init_board()
        chess_board = self.game.numpy_to_board(init_board)
        numpy_board = self.game.board_to_numpy(chess_board)
        self.assertTrue(np.array_equal(init_board, numpy_board))

    def test_move_conversions(self):
        move = chess.Move.from_uci("e2e4")
        action = self.game.move_to_action(move)
        converted_move = self.game.action_to_move(action)
        self.assertEqual(move, converted_move)

    def test_promotion_move(self):
        move = chess.Move.from_uci("e7e8q")
        action = self.game.move_to_action(move)
        converted_move = self.game.action_to_move(action)
        self.assertEqual(move, converted_move)

if __name__ == '__main__':
    unittest.main()
