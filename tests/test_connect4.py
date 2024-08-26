import unittest
import numpy as np
from games.connect4 import Connect4Game
from games.game_utils.connect4_utils import Board

class TestConnect4Game(unittest.TestCase):
    def setUp(self):
        self.game = Connect4Game()

    def test_init(self):
        self.assertEqual(self.game.height, 6)
        self.assertEqual(self.game.width, 7)
        self.assertEqual(self.game.win_length, 4)

    def test_get_init_board(self):
        board = self.game.get_init_board()
        self.assertEqual(board.shape, (6, 7))
        self.assertTrue(np.all(board == 0))

    def test_get_board_size(self):
        self.assertEqual(self.game.get_board_size(), (6, 7))

    def test_get_action_size(self):
        self.assertEqual(self.game.get_action_size(), 7)

    def test_get_next_state(self):
        board = self.game.get_init_board()
        next_board, next_player = self.game.get_next_state(board, 1, 3)
        self.assertEqual(next_board[5][3], 1)
        self.assertEqual(next_player, -1)

    def test_get_valid_moves(self):
        board = self.game.get_init_board()
        valid_moves = self.game.get_valid_moves(board, 1)
        self.assertEqual(valid_moves, [1, 1, 1, 1, 1, 1, 1])

    def test_get_game_ended(self):
        board = self.game.get_init_board()
        self.assertEqual(self.game.get_game_ended(board, 1), 0)

        # Test horizontal win
        board[0][0:4] = 1
        self.assertEqual(self.game.get_game_ended(board, 1), 1)

        # Test vertical win
        board = self.game.get_init_board()
        board[0:4, 0] = -1
        self.assertEqual(self.game.get_game_ended(board, -1), 1)

        # Test diagonal win
        board = self.game.get_init_board()
        for i in range(4):
            board[i, i] = 1
        self.assertEqual(self.game.get_game_ended(board, 1), 1)

    def test_get_canonical_form(self):
        board = np.random.randint(-1, 2, size=(6, 7))
        canonical_board = self.game.get_canonical_form(board, 1)
        self.assertTrue(np.array_equal(canonical_board, board))
        canonical_board = self.game.get_canonical_form(board, -1)
        self.assertTrue(np.array_equal(canonical_board, -board))

    def test_get_symmetries(self):
        board = np.random.randint(-1, 2, size=(6, 7))
        pi = np.random.rand(7)
        symmetries = self.game.get_symmetries(board, pi)
        self.assertEqual(len(symmetries), 2)
        self.assertTrue(np.array_equal(symmetries[0][0], board))
        self.assertTrue(np.array_equal(symmetries[0][1], pi))
        self.assertTrue(np.array_equal(symmetries[1][0], np.fliplr(board)))
        self.assertTrue(np.array_equal(symmetries[1][1], np.flipud(pi)))

    def test_string_representation(self):
        board = self.game.get_init_board()
        self.assertEqual(self.game.string_representation(board), board.tobytes())

if __name__ == '__main__':
    unittest.main()

