import unittest
import torch
from games.tictactoe import TicTacToeGame
from networks.tictactoe_resnet import NNetWrapper
from mcts import MCTS
from utils import dotdict

class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.game = TicTacToeGame()
        self.args = dotdict({
            'num_parallel_envs': 2,
            'max_nodes': 1000,
            'num_mcts_sims': 10,
            'cpuct': 1.0,
            'distributed': False,
            'num_channels': 256,
            'num_res_blocks': 19,
            'lr': 0.001,
            'weight_decay': 0.0001,
            'epochs': 10,
            'batch_size': 64,
            'local_rank': 0
        })
        self.nnet = NNetWrapper(self.game, self.args)
        self.mcts = MCTS(self.game, self.nnet, self.args)

    def test_initialization(self):
        self.assertEqual(self.mcts.num_envs, 2)
        self.assertEqual(self.mcts.action_size, 10)  # 3x3 board + pass action
        self.assertEqual(self.mcts.max_nodes, 1000)

    def test_get_action_prob(self):
        board = self.game.get_init_board()
        canonical_boards = torch.stack([board] * 2)
        probs = self.mcts.get_action_prob(canonical_boards)
        
        self.assertEqual(probs.shape, (2, 10))
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.tensor([1.0, 1.0])))

    def test_search(self):
        board = self.game.get_init_board()
        canonical_boards = torch.stack([board] * 2)
        self.mcts.search(canonical_boards)
        # Check if root node has been expanded
        self.assertTrue(torch.all(self.mcts.Ns[:, 0] > 0))
        # Check if Q-values are within valid range
        self.assertTrue(torch.all(self.mcts.Qsa[:, 0] >= -1))
        self.assertTrue(torch.all(self.mcts.Qsa[:, 0] <= 1))

    def test_uct_scores(self):
        board = self.game.get_init_board()
        canonical_boards = torch.stack([board] * 2)
        self.mcts.search(canonical_boards)  # Perform one search to populate the tree
        uct_scores = self.mcts._uct_scores(torch.zeros(2, dtype=torch.long))
        self.assertEqual(uct_scores.shape, (2, 10))
        self.assertTrue(torch.all(torch.isfinite(uct_scores)))

    def test_expand(self):
        board = self.game.get_init_board()
        canonical_boards = torch.stack([board] * 2)
        s = torch.zeros(2, dtype=torch.long)
        v = self.mcts._expand(canonical_boards, s)
        self.assertEqual(v.shape, (2,))
        self.assertTrue(torch.all(self.mcts.Ps[:, 0] >= 0))
        self.assertTrue(torch.all(self.mcts.Ps[:, 0] <= 1))
        self.assertTrue(torch.allclose(self.mcts.Ps[:, 0].sum(dim=1), torch.tensor([1.0, 1.0])))

    def test_backpropagate(self):
        s = torch.zeros(2, dtype=torch.long)
        a = torch.tensor([0, 1])
        v = torch.tensor([0.5, -0.5])
        self.mcts._backpropagate(s, a, v)
        self.assertEqual(self.mcts.Ns[:, 0].tolist(), [1, 1])
        self.assertEqual(self.mcts.Nsa[:, 0, 0].tolist(), [1, 0])
        self.assertEqual(self.mcts.Nsa[:, 0, 1].tolist(), [0, 1])
        self.assertEqual(self.mcts.Qsa[0, 0, 0].item(), 0.5)
        self.assertEqual(self.mcts.Qsa[1, 0, 1].item(), -0.5)

if __name__ == '__main__':
    unittest.main()