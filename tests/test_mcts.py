import unittest
import torch
from games.tictactoe import TicTacToeGame
from networks.tictactoe_resnet import NNetWrapper
from mcts import MCTS
from utils import dotdict

class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.game = TicTacToeGame()
        self.nnet = NNetWrapper(self.game)
        self.args = dotdict({
            'num_mcts_sims': 100,
            'cpuct': 1.0,
            'num_parallel_envs': 2,
            'max_nodes': 1000
        })
        self.mcts = MCTS(self.game, self.nnet, self.args)

    def test_initialization(self):
        self.assertEqual(self.mcts.num_envs, 2)
        self.assertEqual(self.mcts.action_size, self.game.get_action_size())
        self.assertEqual(self.mcts.max_nodes, 1000)

    def test_get_action_prob(self):
        board = self.game.get_init_board()
        canonical_board = self.game.get_canonical_form(board, 1)
        canonical_boards = torch.stack([torch.from_numpy(canonical_board)] * 2)
        
        probs = self.mcts.get_action_prob(canonical_boards)
        
        self.assertEqual(probs.shape, (2, self.game.get_action_size()))
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.tensor([1.0, 1.0])))

    def test_search(self):
        board = self.game.get_init_board()
        canonical_board = self.game.get_canonical_form(board, 1)
        canonical_boards = torch.stack([torch.from_numpy(canonical_board)] * 2)
        
        self.mcts.search(canonical_boards)
        
        # Check if root node has been expanded
        self.assertTrue(torch.all(self.mcts.Ns[:, 0] > 0))
        
        # Check if Q-values are within valid range
        self.assertTrue(torch.all(self.mcts.Qsa[:, 0] >= -1))
        self.assertTrue(torch.all(self.mcts.Qsa[:, 0] <= 1))

    def test_uct_scores(self):
        board = self.game.get_init_board()
        canonical_board = self.game.get_canonical_form(board, 1)
        canonical_boards = torch.stack([torch.from_numpy(canonical_board)] * 2)
        
        self.mcts.search(canonical_boards)  # Perform one search to populate the tree
        
        uct_scores = self.mcts._uct_scores(torch.zeros(2, dtype=torch.long))
        
        self.assertEqual(uct_scores.shape, (2, self.game.get_action_size()))
        self.assertTrue(torch.all(torch.isfinite(uct_scores)))

    def test_expand(self):
        board = self.game.get_init_board()
        canonical_board = self.game.get_canonical_form(board, 1)
        canonical_boards = torch.stack([torch.from_numpy(canonical_board)] * 2)
        
        v = self.mcts._expand(canonical_boards, torch.zeros(2, dtype=torch.long))
        
        self.assertEqual(v.shape, (2,))
        self.assertTrue(torch.all(v >= -1) and torch.all(v <= 1))
        self.assertTrue(torch.all(self.mcts.Ps[:, 0].sum(dim=1).isclose(torch.tensor([1.0, 1.0]))))

    def test_backpropagate(self):
        s = torch.zeros(2, dtype=torch.long)
        a = torch.tensor([0, 1])
        v = torch.tensor([0.5, -0.5])
        
        initial_q = self.mcts.Qsa[torch.arange(2), s, a].clone()
        initial_n = self.mcts.Nsa[torch.arange(2), s, a].clone()
        
        self.mcts._backpropagate(s, a, v)
        
        expected_q = (initial_n * initial_q + v) / (initial_n + 1)
        self.assertTrue(torch.allclose(self.mcts.Qsa[torch.arange(2), s, a], expected_q))
        self.assertTrue(torch.all(self.mcts.Nsa[torch.arange(2), s, a] == initial_n + 1))
        self.assertTrue(torch.all(self.mcts.Ns[torch.arange(2), s] == initial_n.sum() + 2))

if __name__ == '__main__':
    unittest.main()