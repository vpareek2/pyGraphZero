import unittest
import torch
import numpy as np
from self_play import SelfPlay
from networks.tictactoe_resnet import NNetWrapper
from mcts import MCTS
from games.tictactoe import TicTacToeGame

class MockArgs:
    def __init__(self):
        self.numIters = 2
        self.numEps = 4
        self.tempThreshold = 15
        self.updateThreshold = 0.6
        self.maxlenOfQueue = 200000
        self.numMCTSSims = 25
        self.arenaCompare = 40
        self.cpuct = 1
        self.checkpoint = './temp/'
        self.load_model = False
        self.load_folder_file = ('./temp/', 'best.pth.tar')
        self.numItersForTrainExamplesHistory = 20
        self.distributed = False
        self.num_parallel_envs = 2
        self.max_nodes = 1000
        self.lr = 0.001
        self.dropout_rate = 0.3
        self.epochs = 10
        self.batch_size = 64
        self.num_channels = 512
        self.num_res_blocks = 19
        self.l2_regularization = 1e-4
        self.num_mcts_sims = 25  # Add this line

class TestSelfPlay(unittest.TestCase):
    def setUp(self):
        self.game = TicTacToeGame()
        self.args = MockArgs()
        self.nnet = NNetWrapper(self.game, self.args)
        self.self_play = SelfPlay(self.game, self.nnet, self.args)

    def test_init(self):
        self.assertIsInstance(self.self_play.mcts, MCTS)
        self.assertEqual(len(self.self_play.trainExamplesHistory), 0)
        self.assertFalse(self.self_play.skipFirstSelfPlay)

    def test_execute_episode(self):
        examples = self.self_play.executeEpisode()
        self.assertIsInstance(examples, list)
        self.assertTrue(len(examples) > 0)
        for example in examples:
            self.assertEqual(len(example), 3)
            self.assertIsInstance(example[0], torch.Tensor)  # board
            self.assertIsInstance(example[1], torch.Tensor)  # pi
            self.assertIsInstance(example[2], float)         # v

    def test_learn(self):
        # This is a basic test to ensure the learn method runs without errors
        try:
            self.self_play.learn()
        except Exception as e:
            self.fail(f"learn() raised {type(e).__name__} unexpectedly: {str(e)}")

    def test_get_action_prob(self):
        board = self.game.get_init_board()
        temp = 1
        pi = self.self_play.mcts.get_action_prob(board, temp)
        self.assertIsInstance(pi, np.ndarray)
        self.assertEqual(len(pi), self.game.get_action_size())
        self.assertAlmostEqual(np.sum(pi), 1.0, places=6)

    def test_save_load_train_examples(self):
        # Generate some example data
        examples = [self.self_play.executeEpisode() for _ in range(2)]
        self.self_play.trainExamplesHistory = examples

        # Save examples
        self.self_play.saveTrainExamples(0)

        # Clear examples
        self.self_play.trainExamplesHistory = []

        # Load examples
        self.self_play.loadTrainExamples()

        # Check if examples are loaded correctly
        self.assertEqual(len(self.self_play.trainExamplesHistory), 2)
        for loaded_examples in self.self_play.trainExamplesHistory:
            self.assertIsInstance(loaded_examples, list)
            self.assertTrue(len(loaded_examples) > 0)

if __name__ == '__main__':
    unittest.main()