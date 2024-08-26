import unittest
import numpy as np
from games.chess import ChessGame
import chess

class TestChessGame(unittest.TestCase):
    def setUp(self):
        self.game = ChessGame()

    def test_init(self):
        self.assertIsInstance(self.game.board, chess.Board)

    def test_get_init_board(self):
        board = self.game.get_init_board()
        self.assertEqual(board.shape, (8, 8, 12))
        self.assertEqual(np.sum(board), 32)  # 32 pieces on initial board

    def test_get_board_size(self):
        self.assertEqual(self.game.get_board_size(), (8, 8))

    def test_get_action_size(self):
        self.assertEqual(self.game.get_action_size(), 8 * 8 * 73)

    def test_get_next_state(self):
        board = self.game.get_init_board()
        action = self.game.move_to_action(chess.Move.from_uci("e2e4"))
        next_board, next_player = self.game.get_next_state(board, 1, action)
        
        self.assertEqual(next_player, -1)
        self.assertEqual(np.sum(next_board), 32)  # No piece captured
        self.assertEqual(np.argmax(next_board[3, 4]), 5)  # White pawn moved to e4

    def test_get_valid_moves(self):
        board = self.game.get_init_board()
        valid_moves = self.game.get_valid_moves(board, 1)
        self.assertEqual(np.sum(valid_moves), 20)  # 20 valid moves in initial position

    def test_get_game_ended(self):
        board = self.game.get_init_board()
        self.assertEqual(self.game.get_game_ended(board, 1), 0)

        # Test checkmate
        fen = "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        board = self.game.board_to_numpy()
        self.assertEqual(self.game.get_game_ended(board, 1), -1)

        # Test stalemate
        fen = "5bnr/4p1pq/4Qpkr/7p/7P/4P3/PPPP1PP1/RNB1KBNR b KQ - 2 10"
        board = self.game.board_to_numpy()
        self.assertAlmostEqual(self.game.get_game_ended(board, -1), 1e-4)

    def test_get_canonical_form(self):
        board = self.game.get_init_board()
        canonical_board = self.game.get_canonical_form(board, 1)
        self.assertTrue(np.array_equal(canonical_board, board))
        
        flipped_board = self.game.get_canonical_form(board, -1)
        self.assertTrue(np.array_equal(flipped_board, np.flip(board, axis=0)))

    def test_get_symmetries(self):
        board = self.game.get_init_board()
        pi = np.random.rand(self.game.get_action_size())
        symmetries = self.game.get_symmetries(board, pi)
        self.assertEqual(len(symmetries), 1)
        self.assertTrue(np.array_equal(symmetries[0][0], board))
        self.assertTrue(np.array_equal(symmetries[0][1], pi))

    def test_string_representation(self):
        board = self.game.get_init_board()
        self.assertIsInstance(self.game.string_representation(board), str)

    def test_action_to_move_and_move_to_action(self):
        move = chess.Move.from_uci("e2e4")
        action = self.game.move_to_action(move)
        reconstructed_move = self.game.action_to_move(action)
        self.assertEqual(move, reconstructed_move)

    def test_board_to_numpy_and_numpy_to_board(self):
        original_board = chess.Board()
        numpy_board = self.game.board_to_numpy()
        reconstructed_board = self.game.numpy_to_board(numpy_board)
        self.assertEqual(original_board, reconstructed_board)

if __name__ == '__main__':
    unittest.main()
