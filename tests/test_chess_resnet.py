import unittest
import torch
import numpy as np
from networks.chess_resnet import ResBlock, ChessResNet, NNetWrapper
from games.chess import ChessGame

class TestChessResNet(unittest.TestCase):
    def setUp(self):
        self.game = ChessGame()
        self.args = type('Args', (), {
            'num_channels': 256,
            'num_res_blocks': 19,
            'dropout_rate': 0.3,
            'lr': 0.001,
            'l2_regularization': 0.0001,
            'distributed': False,
            'batch_size': 32,
            'epochs': 10
        })()
        self.nnet = NNetWrapper(self.game, self.args)

    def test_resblock(self):
        block = ResBlock(256)
        x = torch.randn(1, 256, 8, 8)  # (batch_size, channels, height, width)
        out = block(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_chess_resnet_forward(self):
        model = ChessResNet(self.game, self.args)
        x = torch.randn(1, 12, 8, 8)  # (batch_size, input_channels, height, width)
        pi, v = model(x)
        self.assertEqual(pi.shape, (1, self.game.get_action_size()))
        self.assertEqual(v.shape, (1, 1))
        self.assertFalse(torch.isnan(pi).any())
        self.assertFalse(torch.isnan(v).any())

    def test_nnet_predict(self):
        board = np.random.randint(0, 2, size=(8, 8, 12))  # (height, width, channels)
        pi, v = self.nnet.predict(board)
        self.assertEqual(pi.shape, (self.game.get_action_size(),))
        self.assertIsInstance(v, float)
        self.assertTrue(-1 <= v <= 1)  # Changed from 0 <= v <= 1

    def test_nnet_train(self):
        examples = [
            (np.random.randint(0, 2, size=(8, 8, 12)),
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
        board = np.random.randint(0, 2, size=(8, 8, 12))
        pi1, v1 = self.nnet.predict(board)
        pi2, v2 = new_nnet.predict(board)
        np.testing.assert_array_almost_equal(pi1, pi2, decimal=5)
        self.assertAlmostEqual(v1, v2, places=5)

    def test_data_preparation(self):
        examples = [
            (np.random.randint(0, 2, size=(8, 8, 12)),
             np.random.rand(self.game.get_action_size()),
             np.random.rand())
            for _ in range(100)
        ]
        train_data, val_data = self.nnet.prepare_data(examples)
        self.assertIsInstance(train_data, torch.utils.data.TensorDataset)
        self.assertIsInstance(val_data, torch.utils.data.TensorDataset)
        self.assertTrue(len(train_data) + len(val_data) == len(examples))

    def test_data_loader(self):
        examples = [
            (np.random.randint(0, 2, size=(8, 8, 12)),
             np.random.rand(self.game.get_action_size()),
             np.random.rand())
            for _ in range(100)
        ]
        train_data, _ = self.nnet.prepare_data(examples)
        data_loader = self.nnet.get_data_loader(train_data)
        self.assertIsInstance(data_loader, torch.utils.data.DataLoader)
        self.assertEqual(data_loader.batch_size, self.args.batch_size)

    def test_preprocess_board(self):
        board_np = np.random.randint(0, 2, size=(8, 8, 12))
        board_tensor = torch.from_numpy(board_np).float()

        # Test numpy input
        processed_np = self.nnet.preprocess_board(board_np)
        self.assertIsInstance(processed_np, torch.Tensor)
        self.assertEqual(processed_np.shape, (1, 12, 8, 8))

        # Test tensor input
        processed_tensor = self.nnet.preprocess_board(board_tensor)
        self.assertIsInstance(processed_tensor, torch.Tensor)
        self.assertEqual(processed_tensor.shape, (1, 12, 8, 8))

    def test_optimizer_and_scheduler(self):
        self.assertIsInstance(self.nnet.optimizer, torch.optim.Adam)
        self.assertIsInstance(self.nnet.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

if __name__ == '__main__':
    unittest.main()