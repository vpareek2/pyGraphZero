import unittest
import torch
import numpy as np
from networks.tictactoe_resnet import TicTacToeResNet, NNetWrapper

class MockGame:
    def get_board_size(self):
        return (3, 3)
    
    def get_action_size(self):
        return 9

class MockArgs:
    def __init__(self):
        self.num_channels = 32
        self.num_res_blocks = 2
        self.dropout_rate = 0.3
        self.distributed = False
        self.lr = 0.001
        self.l2_regularization = 0.0001
        self.epochs = 10
        self.batch_size = 64

class TestTicTacToeResNet(unittest.TestCase):
    def setUp(self):
        self.game = MockGame()
        self.args = MockArgs()
        self.model = TicTacToeResNet(self.game, self.args)

    def test_model_output_shape(self):
        input_tensor = torch.randn(1, 1, 3, 3)
        pi, v = self.model(input_tensor)
        self.assertEqual(pi.shape, (1, 9))
        self.assertEqual(v.shape, (1,))

    def test_model_forward_pass(self):
        input_tensor = torch.randn(32, 1, 3, 3)
        pi, v = self.model(input_tensor)
        self.assertEqual(pi.shape, (32, 9))
        self.assertEqual(v.shape, (32,))

class TestNNetWrapper(unittest.TestCase):
    def setUp(self):
        self.game = MockGame()
        self.args = MockArgs()
        self.wrapper = NNetWrapper(self.game, self.args)

    def test_predict(self):
        board = np.random.randn(3, 3).astype(np.float32)
        pi, v = self.wrapper.predict(board)
        self.assertEqual(pi.shape, (9,))
        self.assertIsInstance(v, float)

    def test_train(self):
        examples = [
            (np.random.randn(3, 3).astype(np.float32), np.random.rand(9), np.random.rand())
            for _ in range(100)
        ]
        # Convert examples to tensors
        tensor_examples = [
            (torch.FloatTensor(board), torch.FloatTensor(pi), torch.FloatTensor([v]))
            for board, pi, v in examples
        ]
        self.wrapper.train(tensor_examples)

    def test_save_load_checkpoint(self):
        # Save the initial weights
        self.wrapper.save_checkpoint(folder='test_checkpoint', filename='test.pth.tar')
        
        # Change the weights
        for param in self.wrapper.nnet.parameters():
            param.data = torch.randn_like(param.data)
        
        # Load the initial weights
        self.wrapper.load_checkpoint(folder='test_checkpoint', filename='test.pth.tar')
        
        # Check if the weights are restored
        board = np.random.randn(3, 3).astype(np.float32)
        pi1, v1 = self.wrapper.predict(board)
        
        # Save and load again
        self.wrapper.save_checkpoint(folder='test_checkpoint', filename='test2.pth.tar')
        self.wrapper.load_checkpoint(folder='test_checkpoint', filename='test2.pth.tar')
        
        # Check if predictions are the same
        pi2, v2 = self.wrapper.predict(board)
        np.testing.assert_array_almost_equal(pi1, pi2, decimal=6)
        self.assertAlmostEqual(v1, v2, places=6)

if __name__ == '__main__':
    unittest.main()