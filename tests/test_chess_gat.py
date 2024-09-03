import unittest
import torch
import numpy as np
from networks.chess_gat import GATLayer, ChessGAT, NNetWrapper
from games.chess import ChessGame

class TestChessGAT(unittest.TestCase):
    def setUp(self):
        self.game = ChessGame()
        self.args = type('Args', (), {
            'num_channels': 256,
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
        layer = GATLayer(12, 256, num_heads=4)
        x = torch.randn(64, 12)  # 64 nodes (8x8 board), 12 features per node
        edge_index = torch.randint(0, 64, (2, 512))  # Random edges for testing
        out = layer(x, edge_index)
        self.assertEqual(out.shape, (64, 256 * 4))  # 64 nodes, 256 * 4 features per node
        self.assertFalse(torch.isnan(out).any())

    def test_chessgat_forward(self):
        model = ChessGAT(self.game, self.args)
        x = torch.randn(1, 12, 8, 8)  # (batch_size, channels, height, width)
        pi, v = model(x)
        self.assertEqual(pi.shape, (1, self.game.get_action_size()))
        self.assertEqual(v.shape, (1, 1))
        self.assertFalse(torch.isnan(pi).any())
        self.assertFalse(torch.isnan(v).any())

    def test_nnet_predict(self):
        board = np.random.randint(0, 2, size=(12, 8, 8))  # (channels, height, width)
        pi, v = self.nnet.predict(board)
        self.assertEqual(pi.shape, (self.game.get_action_size(),))
        self.assertIsInstance(v, float)
        self.assertTrue(-1 <= v <= 1)

    def test_nnet_train(self):
        examples = [
            (np.random.randint(0, 2, size=(12, 8, 8)),
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
        board = np.random.randint(0, 2, size=(12, 8, 8))
        pi1, v1 = self.nnet.predict(board)
        pi2, v2 = new_nnet.predict(board)
        np.testing.assert_array_almost_equal(pi1, pi2, decimal=5)
        self.assertAlmostEqual(v1, v2, places=5)

    def test_board_to_graph(self):
        model = ChessGAT(self.game, self.args)
        board = torch.randn(1, 12, 8, 8)
        x, edge_index = model._board_to_graph(board)
        self.assertEqual(x.shape, (64, 12))  # 64 nodes, 12 features per node
        self.assertEqual(edge_index.shape[0], 2)  # 2 rows for source and target nodes
        self.assertTrue(edge_index.shape[1] > 0)  # At least some edges

    def test_data_augmentation(self):
        example = (np.random.randint(0, 2, size=(12, 8, 8)),
                   np.random.rand(self.game.get_action_size()),
                   np.random.rand())
        augmented = self.nnet.augment_examples([example])
        self.assertEqual(len(augmented), 2)  # Original + horizontal flip

    def test_validation(self):
        examples = [
            (np.random.randint(0, 2, size=(12, 8, 8)),
             np.random.rand(self.game.get_action_size()),
             np.random.rand())
            for _ in range(10)
        ]
        val_loss = self.nnet.validate(examples)
        self.assertIsInstance(val_loss, float)
        self.assertTrue(val_loss >= 0)

    def test_optimizer_and_scheduler(self):
        self.assertIsInstance(self.nnet.optimizer, torch.optim.Adam)
        self.assertIsInstance(self.nnet.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

if __name__ == '__main__':
    unittest.main()