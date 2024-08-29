import unittest
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import wandb

from games.connect4 import Connect4Game
from networks.connect4_resnet import Connect4ResNet, NNetWrapper

class TestConnect4ResNet(unittest.TestCase):
    def setUp(self):
        self.game = Connect4Game()
        self.args = type('Args', (), {
            'num_channels': 32,
            'num_res_blocks': 2,
            'lr': 0.001,
            'l2_regularization': 0.0001,
            'dropout_rate': 0.3,
            'distributed': False,
            'local_rank': 0,
            'batch_size': 32,
            'epochs': 10
        })()
        self.model = Connect4ResNet(self.game, self.args)
        self.wrapper = NNetWrapper(self.game, self.args)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, Connect4ResNet)
        self.assertEqual(self.model.board_x, 6)
        self.assertEqual(self.model.board_y, 7)
        self.assertEqual(self.model.action_size, 7)  # 7 columns for Connect4

    def test_forward_pass(self):
        board = torch.randn(1, 1, 6, 7)
        pi, v = self.model(board)
        self.assertEqual(pi.shape, (1, 7))
        self.assertEqual(v.shape, (1,))  # Changed from (1, 1) to (1,)

    def test_predict(self):
        # Generate a random board with values -1, 0, or 1
        board = np.random.choice([-1, 0, 1], size=(6, 7))
        pi, v = self.wrapper.predict(board)
        self.assertEqual(pi.shape, (1, 7))
        self.assertIsInstance(v, torch.Tensor)
        self.assertEqual(v.shape, (1, 1))

    def test_loss_functions(self):
        targets = torch.randn(10, 7)
        outputs = torch.randn(10, 7)
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

    def test_augment_examples(self):
        board = np.array([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 1, -1, 0, 0, 0],
                          [0, 1, -1, 1, -1, 0, 0]])
        pi = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2])  # 7 actions
        v = 0.5
        augmented = self.wrapper.augment_examples([(board, pi, v)])
        self.assertEqual(len(augmented), 2)  # Original + horizontal flip

        # Check if the augmented policies have the correct shape
        for aug_board, aug_pi, aug_v in augmented:
            self.assertEqual(aug_board.shape, (1, 6, 7))
            self.assertEqual(aug_pi.shape, (7,))
            self.assertEqual(aug_v, v)

    def test_flip_policy(self):
        pi = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        flipped_pi = self.wrapper.flip_policy(pi)
        expected_flipped_pi = np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        np.testing.assert_array_almost_equal(flipped_pi, expected_flipped_pi)

    def test_augment_batch(self):
        batch = torch.randn(2, 1, 6, 7)
        augmented = self.wrapper.augment_batch(batch)
        self.assertEqual(augmented.shape, (4, 1, 6, 7))  # 2 * 2 augmentations

    def test_edge_cases(self):
        # Empty board
        empty_board = np.zeros((6, 7))
        pi_empty, v_empty = self.wrapper.predict(empty_board)
        self.assertEqual(pi_empty.shape, (1, 7))
        self.assertIsInstance(v_empty, torch.Tensor)
        self.assertEqual(v_empty.shape, (1, 1))

        # Fully filled board
        full_board = np.random.choice([-1, 1], size=(6, 7))
        pi_full, v_full = self.wrapper.predict(full_board)
        self.assertEqual(pi_full.shape, (1, 7))
        self.assertIsInstance(v_full, torch.Tensor)
        self.assertEqual(v_full.shape, (1, 1))

    def test_invalid_inputs(self):
        # Invalid board shape
        with self.assertRaises(ValueError):
            invalid_board = np.random.randn(5, 7)
            self.wrapper.predict(invalid_board)

        # Invalid board values
        with self.assertRaises(ValueError):
            invalid_values_board = np.random.choice([0, 1, 2], size=(6, 7))
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
            self.wrapper.predict([[0, 0, 0, 0, 0, 0, 0]] * 6)

        # Test invalid board shape
        with self.assertRaises(ValueError):
            self.wrapper.predict(np.zeros((5, 7)))

        # Test invalid board values
        with self.assertRaises(ValueError):
            self.wrapper.predict(np.random.choice([0, 1, 2], size=(6, 7)))

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
            (np.random.rand(1, 6, 7), np.random.rand(7), np.random.rand())
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
