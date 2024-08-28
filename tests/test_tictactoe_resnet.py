import unittest
import torch
import numpy as np
from networks.tictactoe_resnet import TicTacToeResNet, NNetWrapper
from games.tictactoe import TicTacToeGame

class TestTicTacToeResNet(unittest.TestCase):
    def setUp(self):
        self.game = TicTacToeGame()
        self.args = type('Args', (), {
            'num_channels': 32,
            'num_res_blocks': 2,
            'lr': 0.001,
            'l2_regularization': 0.0001,
            'dropout_rate': 0.3,
            'distributed': False,
            'local_rank': 0
        })()
        self.model = TicTacToeResNet(self.game, self.args)
        self.wrapper = NNetWrapper(self.game, self.args)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, TicTacToeResNet)
        self.assertEqual(self.model.board_x, 3)
        self.assertEqual(self.model.board_y, 3)
        self.assertEqual(self.model.action_size, 10)  # 9 board positions + 1 pass move

    def test_forward_pass(self):
        board = torch.randn(1, 1, 3, 3)
        pi, v = self.model(board)
        self.assertEqual(pi.shape, (1, 10))
        self.assertEqual(v.shape, (1, 1))

    def test_predict(self):
        # Generate a random board with values -1, 0, or 1
        board = np.random.choice([-1, 0, 1], size=(3, 3))
        pi, v = self.wrapper.predict(board)
        self.assertEqual(pi.shape, (1, 10))
        self.assertIsInstance(v, torch.Tensor)
        self.assertEqual(v.shape, (1, 1))

    def test_loss_functions(self):
        targets = torch.randn(10, 10)
        outputs = torch.randn(10, 10)
        loss_pi = self.wrapper.criterion_pi(outputs, targets.argmax(dim=1))
        self.assertIsInstance(loss_pi, torch.Tensor)

        targets = torch.randn(10, 1)
        outputs = torch.randn(10, 1)
        loss_v = self.wrapper.criterion_v(outputs, targets)
        self.assertIsInstance(loss_v, torch.Tensor)

    def test_save_load_checkpoint(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, 'test_model.pth.tar')
            self.wrapper.save_checkpoint(folder=tmpdirname, filename='test_model.pth.tar')
            self.assertTrue(os.path.exists(filepath))

            new_wrapper = NNetWrapper(self.game, self.args)
            new_wrapper.load_checkpoint(folder=tmpdirname, filename='test_model.pth.tar')

            # Check if the loaded model has the same parameters
            for param1, param2 in zip(self.wrapper.nnet.parameters(), new_wrapper.nnet.parameters()):
                self.assertTrue(torch.equal(param1, param2))

    def test_augment_board(self):
        board = np.array([[1, 0, -1], [0, 1, 0], [-1, 0, 1]])
        pi = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 10 actions
        v = 0.5
        augmented = self.wrapper.augment_examples([(board, pi, v)])
        self.assertEqual(len(augmented), 8)  # 4 rotations * 2 (original + flipped)

    def test_augment_batch(self):
        batch = torch.randn(2, 1, 3, 3)
        augmented = self.wrapper.augment_batch(batch)
        self.assertEqual(augmented.shape, (16, 1, 3, 3))  # 2 * 8 augmentations

    def test_edge_cases(self):
        # Empty board
        empty_board = np.zeros((3, 3))
        pi_empty, v_empty = self.wrapper.predict(empty_board)
        self.assertEqual(pi_empty.shape, (1, 10))
        self.assertIsInstance(v_empty, torch.Tensor)
        self.assertEqual(v_empty.shape, (1, 1))

        # Fully filled board
        full_board = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
        pi_full, v_full = self.wrapper.predict(full_board)
        self.assertEqual(pi_full.shape, (1, 10))
        self.assertIsInstance(v_full, torch.Tensor)
        self.assertEqual(v_full.shape, (1, 1))

    def test_invalid_inputs(self):
        # Invalid board shape
        with self.assertRaises(ValueError):
            invalid_board = np.random.randn(4, 4)
            self.wrapper.predict(invalid_board)

        # Invalid board values
        with self.assertRaises(ValueError):
            invalid_values_board = np.array([[2, 0, -1], [0, 1, 0], [-1, 0, 1]])
            self.wrapper.predict(invalid_values_board)

if __name__ == '__main__':
    unittest.main()
