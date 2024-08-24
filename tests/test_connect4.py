import unittest
import numpy as np
from games.connect4 import Connect4Game

class TestConnect4Game(unittest.TestCase):
    def setUp(self):
        self.game = Connect4Game()

    def test_init(self):
        self.assertEqual(self.game.height, 6)
        self.assertEqual(self.game.width, 7)
        self.assertEqual(self.game.win_length, 4)
        game_custom = Connect4Game(5, 6, 3)
        self.assertEqual(game_custom.height, 5)
        self.assertEqual(game_custom.width, 6)
        self.assertEqual(game_custom.win_length, 3)

    def test_get_init_board(self):
        board = self.game.get_init_board()
        self.assertEqual(board.shape, (6, 7))
        self.assertTrue(np.all(board == 0))

    def test_get_board_size(self):
        self.assertEqual(self.game.get_board_size(), (6, 7))

    def test_get_action_size(self):
        self.assertEqual(self.game.get_action_size(), 7)

    def test_get_next_state(self):
        board = np.zeros((6, 7), dtype=np.int32)
        new_board, next_player = self.game.get_next_state(board, 1, 3)
        self.assertEqual(new_board[5][3], 1)
        self.assertEqual(next_player, -1)

    def test_get_valid_moves(self):
        board = np.zeros((6, 7), dtype=np.int32)
        board[:, 2] = 1
        valid_moves = self.game.get_valid_moves(board, 1)
        expected_valid_moves = [1, 1, 0, 1, 1, 1, 1]
        self.assertTrue(np.array_equal(valid_moves, expected_valid_moves))

    def test_get_game_ended_horizontal(self):
        board = np.zeros((6, 7), dtype=np.int32)
        board[5, 0:4] = 1
        self.assertEqual(self.game.get_game_ended(board, 1), 1)

    def test_get_game_ended_vertical(self):
        board = np.zeros((6, 7), dtype=np.int32)
        board[2:6, 0] = -1
        self.assertEqual(self.game.get_game_ended(board, -1), 1)

    def test_get_game_ended_diagonal(self):
        board = np.zeros((6, 7), dtype=np.int32)
        for i in range(4):
            board[i, i] = 1
        self.assertEqual(self.game.get_game_ended(board, 1), 1)

    def test_get_game_ended_draw(self):
        board = np.ones((6, 7), dtype=np.int32)
        board[::2, ::2] = -1
        print(board)  # Add this line to see the board state
        self.assertAlmostEqual(self.game.get_game_ended(board, 1), 1e-4)

    def test_get_game_ended_ongoing(self):
        board = np.zeros((6, 7), dtype=np.int32)
        board[5, 0:3] = 1
        self.assertEqual(self.game.get_game_ended(board, 1), 0)

    def test_get_canonical_form(self):
        board = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, -1, 0, 0, 0],
            [0, 1, -1, 1, -1, 0, 0]
        ], dtype=np.int32)
        canonical_board = self.game.get_canonical_form(board, -1)
        expected_canonical_board = -board
        self.assertTrue(np.array_equal(canonical_board, expected_canonical_board))

    def test_string_representation(self):
        board = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, -1, 0, 0, 0],
            [0, 1, -1, 1, -1, 0, 0]
        ], dtype=np.int32)
        string_rep = self.game.string_representation(board)
        self.assertIsInstance(string_rep, bytes)
        self.assertEqual(len(string_rep), 42 * board.itemsize)

if __name__ == '__main__':
    unittest.main()