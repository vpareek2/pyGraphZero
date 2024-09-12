import unittest
import numpy as np
from games.tictactoe import TicTacToeGame
from networks.tictactoe_resnet import NNetWrapper
from self_play import SelfPlay
import torch

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class DotDict(dict):
    """A dictionary that allows dot notation access to its items."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class TestSelfPlayIntegration(unittest.TestCase):
    def setUp(self):
        self.args = DotDict({
            'numIters': 2,
            'numEps': 4,
            'tempThreshold': 15,
            'updateThreshold': 0.6,
            'maxlenOfQueue': 200000,
            'numMCTSSims': 25,
            'arenaCompare': 40,
            'cpuct': 1,
            'checkpoint': './temp/',
            'load_model': False,
            'load_folder_file': ('./temp/', 'best.pth.tar'),
            'numItersForTrainExamplesHistory': 20,
            'num_channels': 512,
            'num_res_blocks': 3,
            'dropout_rate': 0.3,
            'lr': 0.001,
            'l2_regularization': 0.0001,
            'distributed': False,
            'batch_size': 32,
            'epochs': 10,
            'max_nodes': 1000
        })

    def test_selfplay_integration(self):
        # Initialize game
        game = TicTacToeGame()

        # Initialize neural network
        nnet = NNetWrapper(game, self.args)

        # Initialize self-play
        sp = SelfPlay(game, nnet, self.args)

        try:
            # Run a few iterations of self-play
            sp.learn()
        except Exception as e:
            self.fail(f"Self-play failed with error: {str(e)}")

    def test_mcts_search_cpu(self):
        self._run_mcts_search_test(force_cpu=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_mcts_search_gpu(self):
        self._run_mcts_search_test(force_cpu=False)

    def _run_mcts_search_test(self, force_cpu):
        game = TicTacToeGame()
        
        if force_cpu:
            self.args.cuda = False
            self.args.device = 'cpu'
        else:
            self.args.cuda = True
            self.args.device = 'cuda'
        
        try:
            nnet = NNetWrapper(game, self.args)
            sp = SelfPlay(game, nnet, self.args)

            # Get initial board
            board = game.get_init_board()

            # Perform MCTS search
            action_probs = sp.mcts.get_action_prob(board)
            self.assertEqual(len(action_probs), game.get_action_size())
            self.assertAlmostEqual(np.sum(action_probs), 1.0, places=6)
        except Exception as e:
            print(f"Detailed error in {'CPU' if force_cpu else 'GPU'} test:")
            import traceback
            traceback.print_exc()
            self.fail(f"MCTS search failed with error: {str(e)}")

    def test_neural_network_predict(self):
        game = TicTacToeGame()
        nnet = NNetWrapper(game, self.args)

        # Get initial board
        board = game.get_init_board()

        try:
            # Get prediction from neural network
            pi, v = nnet.predict(board.numpy())
            self.assertEqual(len(pi), game.get_action_size())
            self.assertIsInstance(v, float)
        except Exception as e:
            self.fail(f"Neural network prediction failed with error: {str(e)}")

if __name__ == '__main__':
    unittest.main()
