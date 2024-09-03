import unittest
import torch
import numpy as np
from networks.connect4_gat import Connect4GAT, NNetWrapper, GATLayer
from games.connect4 import Connect4Game

class TestConnect4GAT(unittest.TestCase):
    def setUp(self):
        self.game = Connect4Game()
        self.args = type('Args', (), {
            'num_channels': 64,
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
        layer = GATLayer(3, 64, num_heads=4)
        x = torch.randn(42, 3)  # 42 nodes (6x7 board), 3 features per node
        edge_index = torch.randint(0, 42, (2, 42 * 41))  # Fully connected graph
        out = layer(x, edge_index)
        self.assertEqual(out.shape, (42, 64 * 4))  # 42 nodes, 64 * 4 features per node

    def test_connect4gat_forward(self):
        model = Connect4GAT(self.game, self.args)
        x = torch.randn(1, 3, 6, 7)  # (batch_size, channels, height, width)
        pi, v = model(x)
        self.assertEqual(pi.shape, (1, self.game.get_action_size()))
        self.assertEqual(v.shape, (1, 1))

    def test_nnet_predict(self):
        board = np.random.randint(0, 3, size=(3, 6, 7))  # (channels, height, width)
        pi, v = self.nnet.predict(board)
        self.assertEqual(pi.shape, (self.game.get_action_size(),))
        self.assertIsInstance(v, float)

    def test_nnet_train(self):
        examples = [
            (np.random.randint(0, 3, size=(3, 6, 7)),
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
        board = np.random.randint(0, 3, size=(3, 6, 7))
        pi1, v1 = self.nnet.predict(board)
        pi2, v2 = new_nnet.predict(board)
        np.testing.assert_array_almost_equal(pi1, pi2, decimal=5)
        self.assertAlmostEqual(v1, v2, places=5)

    def test_data_augmentation(self):
        board = np.random.randint(0, 3, size=(3, 6, 7))
        pi = np.random.rand(self.game.get_action_size())
        v = np.random.rand()
        
        example = (board, pi, v)
        augmented = self.nnet.augment_examples([example])
        
        self.assertEqual(len(augmented), 2)  # Original + 1 flip
        
        # Check original
        self.assertTrue(np.array_equal(augmented[0][0], board))
        self.assertTrue(np.array_equal(augmented[0][1], pi))
        self.assertEqual(augmented[0][2], v)
        
        # Check flip
        self.assertTrue(np.array_equal(augmented[1][0], np.fliplr(board)))
        self.assertTrue(np.array_equal(augmented[1][1], np.flip(pi)))
        self.assertEqual(augmented[1][2], v)
        
        # Check shapes and sizes
        for aug_board, aug_pi, aug_v in augmented:
            self.assertEqual(aug_board.shape, (3, 6, 7))
            self.assertEqual(aug_pi.shape, (self.game.get_action_size(),))
            self.assertIsInstance(aug_v, float)

    def test_untrained_nnet_predict(self):
        board = np.zeros((3, 6, 7))  # Empty board
        
        pi, v = self.nnet.predict(board)
        
        # Check policy
        self.assertEqual(pi.shape, (self.game.get_action_size(),))
        self.assertAlmostEqual(np.sum(pi), 1.0, places=6)
        self.assertTrue(np.all(pi >= 0))
        
        # Check value
        self.assertIsInstance(v, float)
        self.assertGreaterEqual(v, -1.0)
        self.assertLessEqual(v, 1.0)
        
if __name__ == '__main__':
    unittest.main()