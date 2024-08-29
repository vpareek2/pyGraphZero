import unittest
import torch
import numpy as np
from networks.connect4_gat import Connect4GAT, NNetWrapper, GATLayer
from games.connect4 import Connect4Game
import torch.nn as nn
import torch.distributed as dist
import wandb

class TestConnect4GAT(unittest.TestCase):
    def setUp(self):
        self.game = Connect4Game()
        self.args = type('Args', (), {
            'num_channels': 32,
            'lr': 0.001,
            'l2_regularization': 0.0001,
            'dropout_prob': 0.3,
            'distributed': False,
            'local_rank': 0,
            'batch_size': 32,
            'epochs': 10
        })()
        self.model = Connect4GAT(self.game, self.args)
        self.wrapper = NNetWrapper(self.game, self.args)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, Connect4GAT)
        self.assertEqual(self.model.board_x, 6)  # Connect4 board width
        self.assertEqual(self.model.board_y, 7)  # Connect4 board height
        self.assertEqual(self.model.action_size, 7)  # Connect4 has 7 possible actions (columns)

    def test_gat_layer(self):
        gat_layer = GATLayer(3, 32, num_heads=4, dropout_prob=0.3)
        self.assertIsInstance(gat_layer, GATLayer)
        self.assertEqual(gat_layer.in_features, 3)
        self.assertEqual(gat_layer.out_features, 32)
        self.assertEqual(gat_layer.num_heads, 4)

    def test_forward_pass(self):
        board = torch.randn(1, 1, 6, 7)  # Connect4 board size
        pi, v = self.model(board)
        self.assertEqual(pi.shape, (1, 7))  # 7 possible actions
        self.assertEqual(v.shape, (1, 1))

    def test_predict(self):
        board = np.random.choice([-1, 0, 1], size=(6, 7))  # Connect4 board size
        pi, v = self.wrapper.predict(board)
        self.assertEqual(pi.shape, (1, 7))  # 7 possible actions
        self.assertIsInstance(v, torch.Tensor)
        self.assertEqual(v.shape, (1, 1))

    def test_loss_functions(self):
        targets = torch.randn(10, 7)  # 7 possible actions
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

            for param1, param2 in zip(self.wrapper.nnet.parameters(), new_wrapper.nnet.parameters()):
                self.assertTrue(torch.equal(param1, param2))

    def test_augment_examples(self):
        board = np.random.choice([-1, 0, 1], size=(6, 7))
        pi = np.random.rand(7)  # 7 possible actions
        v = 0.5
        augmented = self.wrapper.augment_examples([(board, pi, v)])
        self.assertEqual(len(augmented), 3)  # Original + horizontal flip + noise

        for aug_board, aug_pi, aug_v in augmented:
            self.assertEqual(aug_board.shape, (6, 7))
            self.assertEqual(aug_pi.shape, (7,))
            self.assertEqual(aug_v, v)

    def test_augment_batch(self):
        batch_size = 2
        board_size = (6, 7)  # Connect4 board size
        policy_size = 7  # 7 possible actions
        batch = torch.randn(batch_size, 1, *board_size)
        pis = torch.randn(batch_size, policy_size)
        
        augmented_boards, augmented_pis = self.wrapper.augment_batch(batch, pis)
        
        self.assertEqual(augmented_boards.shape[0], batch_size * 2)  # Original + horizontal flip
        self.assertEqual(augmented_pis.shape[0], batch_size * 2)
        self.assertEqual(augmented_pis.shape[1], policy_size)

    def test_edge_cases(self):
        empty_board = np.zeros((6, 7))
        pi_empty, v_empty = self.wrapper.predict(empty_board)
        self.assertEqual(pi_empty.shape, (1, 7))
        self.assertIsInstance(v_empty, torch.Tensor)
        self.assertEqual(v_empty.shape, (1, 1))

        full_board = np.random.choice([-1, 1], size=(6, 7))
        pi_full, v_full = self.wrapper.predict(full_board)
        self.assertEqual(pi_full.shape, (1, 7))
        self.assertIsInstance(v_full, torch.Tensor)
        self.assertEqual(v_full.shape, (1, 1))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            invalid_board = np.random.randn(5, 5)
            self.wrapper.predict(invalid_board)

        with self.assertRaises(ValueError):
            invalid_values_board = np.array([[2, 0, -1, 1, 0, -1, 1]] * 6)
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

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gpu_usage(self):
        self.assertIsInstance(self.wrapper.device, torch.device)
        self.assertTrue(self.wrapper.device.type in ['cuda', 'cpu'])

    def test_early_stopping(self):
        dummy_examples = [
            (np.random.rand(6, 7), np.random.rand(7), np.random.rand())
            for _ in range(100)
        ]
        
        original_validate = self.wrapper.validate
        self.wrapper.validate = lambda x: 1.1
        
        self.wrapper.best_val_loss = 1.0
        self.wrapper.patience = 3
        
        for _ in range(5):
            self.wrapper.train(dummy_examples)
        
        self.assertLess(self.wrapper.wait, 5)
        self.assertGreaterEqual(self.wrapper.wait, 3)
        
        self.wrapper.validate = original_validate

    def test_distributed_setup(self):
        if self.wrapper.args.distributed:
            self.assertIsNotNone(self.wrapper.world_size)
            self.assertIsNotNone(self.wrapper.rank)
            self.assertTrue(dist.is_initialized())

if __name__ == '__main__':
    unittest.main()
