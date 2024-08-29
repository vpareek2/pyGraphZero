import unittest
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import wandb

from games.chess import ChessGame
from networks.chess_resnet import ChessResNet, NNetWrapper

class TestChessResNet(unittest.TestCase):
    def setUp(self):
        self.game = ChessGame()
        self.args = type('Args', (), {
            'num_channels': 256,
            'num_res_blocks': 19,
            'lr': 0.001,
            'l2_regularization': 0.0001,
            'dropout_rate': 0.3,
            'distributed': False,
            'local_rank': 0,
            'batch_size': 32,
            'epochs': 10
        })()
        self.model = ChessResNet(self.game, self.args)
        self.wrapper = NNetWrapper(self.game, self.args)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, ChessResNet)
        self.assertEqual(self.model.board_x, 8)
        self.assertEqual(self.model.board_y, 8)
        self.assertEqual(self.model.action_size, 8 * 8 * 73)  # 8x8 board, 73 possible moves per square

    def test_forward_pass(self):
        board = torch.randn(1, 12, 8, 8)  # 12 channels for chess
        pi, v = self.model(board)
        self.assertEqual(pi.shape, (1, 8 * 8 * 73))
        self.assertEqual(v.shape, (1,))

    def test_predict(self):
        # Generate a random board with values 0 or 1
        board = np.random.choice([0, 1], size=(8, 8, 12))
        pi, v = self.wrapper.predict(board)
        self.assertEqual(pi.shape, (1, 8 * 8 * 73))
        self.assertIsInstance(v, torch.Tensor)
        self.assertEqual(v.shape, (1,))

    def test_loss_functions(self):
        targets = torch.randn(10, 8 * 8 * 73)
        outputs = torch.randn(10, 8 * 8 * 73)
        loss_pi = self.wrapper.criterion_pi(outputs, targets)
        self.assertIsInstance(loss_pi, torch.Tensor)

        targets = torch.randn(10)
        outputs = torch.randn(10)
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

    def test_augment_examples(self):
        board = np.random.choice([0, 1], size=(8, 8, 12))
        pi = np.random.rand(8 * 8 * 73)
        v = 0.5
        augmented = self.wrapper.augment_examples([(board, pi, v)])
        self.assertEqual(len(augmented), 2)  # Original + horizontal flip

        # Check if the augmented policies have the correct shape
        for aug_board, aug_pi, aug_v in augmented:
            self.assertEqual(aug_board.shape, (8, 8, 12))
            self.assertEqual(aug_pi.shape, (8 * 8 * 73,))
            self.assertEqual(aug_v, v)

    def test_flip_policy(self):
        pi = np.random.rand(8 * 8 * 73)
        flipped_pi = self.wrapper.flip_policy(pi)
        self.assertEqual(flipped_pi.shape, (8 * 8 * 73,))
        # Note: You might want to add more specific checks for the flipping logic

    def test_augment_batch(self):
        batch = torch.randn(2, 12, 8, 8)
        augmented = self.wrapper.augment_batch(batch)
        self.assertEqual(augmented.shape, (4, 12, 8, 8))  # 2 * 2 augmentations

    def test_edge_cases(self):
        # Empty board
        empty_board = np.zeros((8, 8, 12))
        pi_empty, v_empty = self.wrapper.predict(empty_board)
        self.assertEqual(pi_empty.shape, (1, 8 * 8 * 73))
        self.assertIsInstance(v_empty, torch.Tensor)
        self.assertEqual(v_empty.shape, (1,))

        # Fully filled board
        full_board = np.random.choice([0, 1], size=(8, 8, 12))
        pi_full, v_full = self.wrapper.predict(full_board)
        self.assertEqual(pi_full.shape, (1, 8 * 8 * 73))
        self.assertIsInstance(v_full, torch.Tensor)
        self.assertEqual(v_full.shape, (1,))

    def test_invalid_inputs(self):
        # Invalid board shape
        with self.assertRaises(ValueError):
            invalid_board = np.random.randn(7, 8, 12)
            self.wrapper.predict(invalid_board)

        # Invalid board values
        with self.assertRaises(ValueError):
            invalid_values_board = np.random.choice([0, 1, 2], size=(8, 8, 12))
            self.wrapper.predict(invalid_values_board)

    def test_optimizer_and_scheduler(self):
        self.assertIsInstance(self.wrapper.optimizer, torch.optim.Adam)
        self.assertIsInstance(self.wrapper.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_criterion(self):
        self.assertIsInstance(self.wrapper.criterion_pi, nn.CrossEntropyLoss)
        self.assertIsInstance(self.wrapper.criterion_v, nn.MSELoss)

    def test_scaler(self):
        from torch.amp import GradScaler
        self.assertIsInstance(self.wrapper.scaler, GradScaler)

    def test_predict_input_validation(self):
        # Test invalid input type
        with self.assertRaises(ValueError):
            self.wrapper.predict([[0, 0, 0, 0, 0, 0, 0, 0]] * 8)

        # Test invalid board shape
        with self.assertRaises(ValueError):
            self.wrapper.predict(np.zeros((7, 8, 12)))

        # Test invalid board values
        with self.assertRaises(ValueError):
            self.wrapper.predict(np.random.choice([0, 1, 2], size=(8, 8, 12)))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gpu_usage(self):
        self.assertIsInstance(self.wrapper.device, torch.device)
        self.assertTrue(self.wrapper.device.type in ['cuda', 'cpu'])

    def test_wandb_initialization(self):
        # This test assumes wandb is initialized in the main process
        if self.wrapper.is_main_process():
            self.assertTrue(wandb.run is not None)

    def test_early_stopping(self):
        # Create some dummy examples
        dummy_examples = [
            (np.random.choice([0, 1], size=(8, 8, 12)), np.random.rand(8 * 8 * 73), np.random.rand())
            for _ in range(100)
        ]

        # Mock the validate method to always return a worse loss
        original_validate = self.wrapper.validate
        self.wrapper.validate = lambda x: 1.1

        # Set initial best_val_loss
        self.wrapper.best_val_loss = 1.0
        self.wrapper.patience = 3

        # Train for a few epochs
        for _ in range(5):
            self.wrapper.train(dummy_examples)

        # Check if training stopped early
        self.assertLess(self.wrapper.wait, 5)
        self.assertGreaterEqual(self.wrapper.wait, 3)

        # Restore the original validate method
        self.wrapper.validate = original_validate

    def test_distributed_setup(self):
        if self.wrapper.args.distributed:
            self.assertIsNotNone(self.wrapper.world_size)
            self.assertIsNotNone(self.wrapper.rank)
            self.assertTrue(dist.is_initialized())

if __name__ == '__main__':
    unittest.main()
