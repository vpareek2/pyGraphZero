import unittest
import torch
from networks.tictactoe_resnet import TicTacToeResNet, NNetWrapper
from games.tictactoe import TicTacToeGame
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler
from torch.optim import lr_scheduler
import numpy as np

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
            'local_rank': 0,
            'batch_size': 64,
            'epochs': 10
        })()
        self.model = TicTacToeResNet(self.game, self.args)
        self.wrapper = NNetWrapper(self.game, self.args)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, TicTacToeResNet)
        self.assertEqual(self.model.board_x, 3)
        self.assertEqual(self.model.board_y, 3)
        self.assertEqual(self.model.action_size, 9)

    def test_forward_pass(self):
        board = torch.randn(1, 1, 3, 3)
        pi, v = self.model(board)
        self.assertEqual(pi.shape, (1, 9))
        self.assertEqual(v.shape, (1,))

    def test_predict(self):
        board = self.game.get_init_board()
        pi, v = self.wrapper.predict(board)
        self.assertEqual(pi.shape, (self.game.get_action_size(),))
        self.assertIsInstance(v, float)

    def test_valid_probability_distribution(self):
        board = self.game.get_init_board()
        pi, _ = self.wrapper.predict(board)
        pi_tensor = torch.tensor(pi)  # Convert NumPy array to PyTorch tensor
        self.assertAlmostEqual(pi_tensor.sum().item(), 1.0, places=6)
        self.assertTrue(np.all(pi >= 0) and np.all(pi <= 1))

    def test_different_inputs(self):
        boards = [
            self.game.get_init_board(),
            torch.tensor([[1, 0, -1], [0, 1, 0], [-1, 0, 1]], dtype=torch.float32),
            torch.tensor([[1, 1, 1], [-1, -1, 0], [0, 0, 0]], dtype=torch.float32)
        ]
        for board in boards:
            pi, v = self.wrapper.predict(board)
            self.assertEqual(pi.shape, (self.game.get_action_size(),))
            self.assertIsInstance(v, float)

    def test_loss_functions(self):
        self.assertIsInstance(self.wrapper.criterion_pi, nn.CrossEntropyLoss)
        self.assertIsInstance(self.wrapper.criterion_v, nn.MSELoss)

    def test_save_load_checkpoint(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, 'test_model.pth.tar')
            self.wrapper.save_checkpoint(folder=tmpdirname, filename='test_model.pth.tar')
            self.assertTrue(os.path.exists(filepath))

            new_wrapper = NNetWrapper(self.game, self.args)
            new_wrapper.load_checkpoint(folder=tmpdirname, filename='test_model.pth.tar')

            # Compare state_dicts
            original_state_dict = self.wrapper.nnet.state_dict()
            loaded_state_dict = new_wrapper.nnet.state_dict()

            for key in original_state_dict:
                self.assertTrue(torch.equal(original_state_dict[key], loaded_state_dict[key]))

            # Check optimizer state
            self.assertEqual(self.wrapper.optimizer.state_dict()['param_groups'],
                             new_wrapper.optimizer.state_dict()['param_groups'])

            # Check scheduler state
            self.assertEqual(self.wrapper.scheduler.state_dict()['patience'],
                             new_wrapper.scheduler.state_dict()['patience'])

            # Check scaler state
            self.assertEqual(self.wrapper.scaler.state_dict()['scale'],
                             new_wrapper.scaler.state_dict()['scale'])

            # Compare outputs of both models
            board = self.game.get_init_board()
            pi1, v1 = self.wrapper.predict(board)
            pi2, v2 = new_wrapper.predict(board)
            self.assertTrue(torch.allclose(torch.tensor(pi1), torch.tensor(pi2)))
            self.assertAlmostEqual(v1, v2)

    def test_prepare_data(self):
        examples = [
            (torch.rand(3, 3), torch.rand(9), torch.rand(1))
            for _ in range(100)
        ]
        train_data, val_data = self.wrapper.prepare_data(examples)
        self.assertIsInstance(train_data, tuple)
        self.assertIsInstance(val_data, tuple)
        self.assertEqual(len(train_data), 3)
        self.assertEqual(len(val_data), 3)

    def test_get_data_loader(self):
        data = torch.utils.data.TensorDataset(
            torch.randn(100, 1, 3, 3),
            torch.randn(100, 9),
            torch.randn(100)
        )
        loader = self.wrapper.get_data_loader(data)
        self.assertIsInstance(loader, torch.utils.data.DataLoader)
        self.assertEqual(loader.batch_size, self.args.batch_size)

    def test_train_epoch(self):
        data = torch.utils.data.TensorDataset(
            torch.randn(100, 1, 3, 3),
            torch.randn(100, 9),
            torch.randn(100)
        )
        loader = self.wrapper.get_data_loader(data)
        loss = self.wrapper.train_epoch(loader)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)  # Loss should be positive

    def test_validate(self):
        val_data = (
            torch.randn(100, 1, 3, 3),
            torch.randn(100, 9),
            torch.randn(100)
        )
        loss = self.wrapper.validate(val_data)
        self.assertIsInstance(loss, float)

    def test_training_improves_loss(self):
        # Generate some random training data
        num_examples = 1000
        boards = torch.randint(-1, 2, (num_examples, 3, 3), dtype=torch.float32)
        pis = torch.rand(num_examples, self.game.get_action_size())
        pis = pis / pis.sum(dim=1, keepdim=True)  # Normalize
        vs = torch.rand(num_examples, 1) * 2 - 1  # Values between -1 and 1, shape (1000, 1)
        examples = list(zip(boards.numpy(), pis.numpy(), vs.squeeze().numpy()))
        
        # Train for a few epochs and check if loss decreases
        train_data = self.wrapper.examples_to_tensors(examples)
        train_loader = self.wrapper.get_data_loader(train_data)
        
        initial_loss = self.wrapper.train_epoch(train_loader)
        self.wrapper.train(examples)
        final_loss = self.wrapper.train_epoch(train_loader)
        self.assertLess(final_loss, initial_loss)

    def test_symmetries(self):
        board = torch.tensor([[1, 0, -1], [0, 1, 0], [-1, 0, 1]], dtype=torch.float32)
        pi = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        v = 0.5
        symmetries = self.wrapper.get_symmetries(board, pi, v)
        self.assertEqual(len(symmetries), 8)  # 4 rotations * 2 (with and without flip)
        # Check if original board is in symmetries
        self.assertTrue(any(torch.equal(board, torch.tensor(sym[0])) for sym in symmetries))

    def test_optimizer_and_scheduler(self):
        self.assertIsInstance(self.wrapper.optimizer, torch.optim.Adam)
        self.assertIsInstance(self.wrapper.scheduler, lr_scheduler.ReduceLROnPlateau)

    def test_scaler(self):
        self.assertIsInstance(self.wrapper.scaler, GradScaler)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gpu_usage(self):
        self.assertIsInstance(self.wrapper.device, torch.device)
        self.assertTrue(self.wrapper.device.type in ['cuda', 'cpu'])

    def test_is_main_process(self):
        self.assertTrue(self.wrapper.is_main_process())

    @unittest.skipIf(not dist.is_available(), "Distributed not available")
    def test_reduce_tensor(self):
        # Initialize distributed environment for testing
        if not dist.is_initialized():
            dist.init_process_group(backend='gloo', init_method='tcp://localhost:12345', rank=0, world_size=1)

        tensor = torch.tensor([1.0, 2.0, 3.0])
        reduced = self.wrapper.reduce_tensor(tensor)
        self.assertTrue(torch.allclose(tensor, reduced))

        # Clean up
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    unittest.main()