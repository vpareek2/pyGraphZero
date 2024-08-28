import unittest
import torch
from games.tictactoe import TicTacToeGame

class TestTicTacToeGame(unittest.TestCase):
    def setUp(self):
        self.game = TicTacToeGame()

    def test_init(self):
        self.assertEqual(self.game.n, 3)
        game_4x4 = TicTacToeGame(4)
        self.assertEqual(game_4x4.n, 4)

    def test_get_init_board(self):
        board = self.game.get_init_board()
        self.assertEqual(board.shape, (3, 3))
        self.assertTrue(torch.all(board == 0))

    def test_get_board_size(self):
        self.assertEqual(self.game.get_board_size(), (3, 3))

    def test_get_action_size(self):
        self.assertEqual(self.game.get_action_size(), 10)  # 3x3 + 1 for pass

    def test_get_next_state(self):
        board = torch.zeros((3, 3))
        new_board, next_player = self.game.get_next_state(board, 1, 4)
        self.assertEqual(new_board[1, 1], 1)
        self.assertEqual(next_player, -1)

    def test_get_valid_moves(self):
        board = torch.tensor([
            [1, 0, -1],
            [0, 1, 0],
            [-1, 0, 0]
        ], dtype=torch.float32)
        valid_moves = self.game.get_valid_moves(board, 1)
        expected_valid_moves = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 1, 1], dtype=torch.float32)
        self.assertTrue(torch.all(valid_moves == expected_valid_moves))

    def test_get_game_ended(self):
        # Test win for player 1
        board_win = torch.tensor([
            [1, 1, 1],
            [0, -1, 0],
            [-1, 0, 0]
        ], dtype=torch.float32)
        self.assertEqual(self.game.get_game_ended(board_win, 1).item(), 1)

        # Test draw
        board_draw = torch.tensor([
            [1, -1, 1],
            [-1, 1, -1],
            [-1, 1, -1]
        ], dtype=torch.float32)
        self.assertAlmostEqual(self.game.get_game_ended(board_draw, 1).item(), 1e-4)

        # Test game not ended
        board_ongoing = torch.tensor([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 0]
        ], dtype=torch.float32)
        self.assertEqual(self.game.get_game_ended(board_ongoing, 1).item(), 0)

    def test_get_canonical_form(self):
        board = torch.tensor([
            [1, 0, -1],
            [0, 1, 0],
            [-1, 0, 0]
        ], dtype=torch.float32)
        canonical_board = self.game.get_canonical_form(board, -1)
        expected_canonical_board = torch.tensor([
            [-1, 0, 1],
            [0, -1, 0],
            [1, 0, 0]
        ], dtype=torch.float32)
        self.assertTrue(torch.all(canonical_board == expected_canonical_board))

    def test_get_symmetries(self):
        board = torch.tensor([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 0]
        ], dtype=torch.float32)
        pi = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0])
        symmetries = self.game.get_symmetries(board, pi)
        self.assertEqual(len(symmetries), 8)  # 4 rotations * 2 (with and without flip)

    def test_string_representation(self):
        board = torch.tensor([
            [1, 0, -1],
            [0, 1, 0],
            [-1, 0, 0]
        ], dtype=torch.float32)
        string_rep = self.game.string_representation(board)
        self.assertIsInstance(string_rep, bytes)
        self.assertEqual(len(string_rep), 9 * board.element_size())

if __name__ == '__main__':
    unittest.main()