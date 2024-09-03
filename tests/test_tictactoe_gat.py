import unittest
import torch
import numpy as np
from networks.tictactoe_gat import TicTacToeGAT, NNetWrapper, GATLayer
from games.tictactoe import TicTacToeGame

class TestTicTacToeGAT(unittest.TestCase):
    def setUp(self):
        self.game = TicTacToeGame()
        self.args = type('Args', (), {
            'num_channels': 32,
            'num_heads': 4,
            'dropout_rate': 0.3,
            'lr': 0.001,
            'l2_regularization': 0.0001,
            'distributed': False,
            'batch_size': 32,
            'epochs': 10
        })()
        self.nnet = NNetWrapper(self.game, self.args)

    def test_gatlayer(self):
        layer = GATLayer(3, 32, num_heads=4)
        x = torch.randn(9, 3)  # 9 nodes (3x3 board), 3 features per node
        edge_index = torch.randint(0, 9, (2, 72))  # Fully connected graph (9 * 8 edges)
        out = layer(x, edge_index)
        self.assertEqual(out.shape, (9, 32 * 4))  # 9 nodes, 32 * 4 features per node

    def test_tictactoegat_forward(self):
        model = TicTacToeGAT(self.game, self.args)
        x = torch.randn(1, 3, 3, 3)  # (batch_size, channels, height, width)
        pi, v = model(x)
        self.assertEqual(pi.shape, (1, self.game.get_action_size()))
        self.assertEqual(v.shape, (1, 1))

    def test_nnet_predict(self):
        board = np.random.randint(0, 3, size=(3, 3, 3))  # (channels, height, width)
        pi, v = self.nnet.predict(board)
        self.assertEqual(pi.shape, (self.game.get_action_size(),))
        self.assertIsInstance(v, float)  # Changed from v[0] to v

    def test_nnet_train(self):
        examples = [
            (np.random.randint(0, 3, size=(3, 3, 3)),
             np.random.rand(self.game.get_action_size()),
             np.random.rand())
            for _ in range(100)
        ]
        self.nnet.train(examples)

    def test_nnet_save_load(self):
        # Save the model
        self.nnet.save_checkpoint(folder='test_checkpoint', filename='test_model.pth.tar')

        # Create a new model and load the saved weights
        new_nnet = NNetWrapper(self.game, self.args)
        new_nnet.load_checkpoint(folder='test_checkpoint', filename='test_model.pth.tar')

        # Compare predictions
        board = np.random.randint(0, 3, size=(3, 3, 3))
        pi1, v1 = self.nnet.predict(board)
        pi2, v2 = new_nnet.predict(board)
        np.testing.assert_array_almost_equal(pi1, pi2, decimal=5)
        self.assertAlmostEqual(v1, v2, places=5)  # Changed from v1[0], v2[0] to v1, v2

    def test_data_augmentation(self):
        board = np.random.randint(0, 3, size=(3, 3, 3))
        pi = np.random.rand(self.game.get_action_size())
        v = np.random.rand()
        
        example = (board, pi, v)
        augmented = self.nnet.augment_examples([example])
        
        self.assertEqual(len(augmented), 5)  # Original + 3 rotations + 1 flip
        
        # Check original
        self.assertTrue(np.array_equal(augmented[0][0], board))
        self.assertTrue(np.array_equal(augmented[0][1], pi))
        self.assertEqual(augmented[0][2], v)
        
        # Check rotations
        for i in range(1, 4):
            self.assertTrue(np.array_equal(augmented[i][0], np.rot90(board, i)))
            self.assertTrue(np.array_equal(augmented[i][1], np.rot90(pi.reshape(3, 3), i).flatten()))
            self.assertEqual(augmented[i][2], v)
        
        # Check flip
        self.assertTrue(np.array_equal(augmented[4][0], np.fliplr(board)))
        self.assertTrue(np.array_equal(augmented[4][1], np.fliplr(pi.reshape(3, 3)).flatten()))
        self.assertEqual(augmented[4][2], v)
        
        # Check shapes and sizes
        for aug_board, aug_pi, aug_v in augmented:
            self.assertEqual(aug_board.shape, (3, 3, 3))
            self.assertEqual(aug_pi.shape, (self.game.get_action_size(),))
            self.assertIsInstance(aug_v, float)

    def test_untrained_nnet_predict(self):
        board = np.zeros((3, 3, 3))  # Empty board
        
        pi, v = self.nnet.predict(board)
        
        # Check policy
        self.assertEqual(pi.shape, (self.game.get_action_size(),))
        self.assertAlmostEqual(np.sum(pi), 1.0, places=6)
        self.assertTrue(np.all(pi >= 0))
        
        # Check value
        self.assertIsInstance(v, float)
        self.assertGreaterEqual(v, -1.0)
        self.assertLessEqual(v, 1.0)
        
        print(f"Untrained model - Policy: {pi}, Value: {v}")

if __name__ == '__main__':
    unittest.main()