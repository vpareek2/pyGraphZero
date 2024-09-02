import unittest
import torch
import numpy as np
from networks.connect4_resnet import Connect4ResNet, NNetWrapper, ResBlock
from games.connect4 import Connect4Game


class TestConnect4ResNet(unittest.TestCase):
    def setUp(self):
        self.game = Connect4Game()
        self.args = type('Args', (), {
            'num_channels': 64,
            'num_res_blocks': 3,
            'dropout_rate': 0.3,
            'lr': 0.001,
            'l2_regularization': 0.0001,
            'distributed': False,
            'batch_size': 32,
            'epochs': 10
        })()
        self.nnet = NNetWrapper(self.game, self.args)

    def test_resblock(self):
        block = ResBlock(64)
        x = torch.randn(1, 64, 6, 7)
        out = block(x)
        self.assertEqual(out.shape, x.shape)

    def test_connect4resnet_forward(self):
        model = Connect4ResNet(self.game, self.args)
        x = torch.randn(1, 1, 6, 7)
        pi, v = model(x)
        self.assertEqual(pi.shape, (1, self.game.get_action_size()))
        self.assertEqual(v.shape, (1,))

    def test_nnet_predict(self):
        board = np.random.randint(0, 3, size=(6, 7))
        pi, v = self.nnet.predict(board)
        self.assertEqual(pi.shape, (self.game.get_action_size(),))
        self.assertIsInstance(v, float)

    def test_nnet_train(self):
        examples = [
            (np.random.randint(0, 3, size=(6, 7)), 
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
        board = np.random.randint(0, 3, size=(6, 7))
        pi1, v1 = self.nnet.predict(board)
        pi2, v2 = new_nnet.predict(board)
        
        np.testing.assert_array_almost_equal(pi1, pi2, decimal=5)
        self.assertAlmostEqual(v1, v2, places=5)

    def test_data_augmentation(self):
        example = (np.random.randint(0, 3, size=(6, 7)), 
                   np.random.rand(self.game.get_action_size()), 
                   np.random.rand())
        augmented = self.nnet.augment_examples([example])
        self.assertEqual(len(augmented), 2)  # Original + vertical flip

if __name__ == '__main__':
    unittest.main()