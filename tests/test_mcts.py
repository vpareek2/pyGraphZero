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
            'lr': 0.001,
            'dropout': 0.3,
            'epochs': 10,
            'batch_size': 64,
            'num_channels': 512,
            'num_res_blocks': 19,
            'weight_decay': 1e-4,
            'distributed': False,
            'local_rank': 0
        })
        self.nnet = NNetWrapper(self.game, self.args)
        self.mcts_args = dotdict({
            'num_mcts_sims': 100,
            'cpuct': 1.0,
            'num_parallel_envs': 2,
            'max_nodes': 1000
        })
        self.mcts = MCTS(self.game, self.nnet, self.mcts_args)

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

    def _expand(self, canonical_boards, s):
        pi, v = self.nnet.predict(canonical_boards)
        self.Ps[torch.arange(self.num_envs), s] = pi.to(self.device)
        valids = self.game.get_valid_moves(canonical_boards, 1)
        self.Ps[torch.arange(self.num_envs), s] *= valids
        sum_Ps_s = self.Ps[torch.arange(self.num_envs), s].sum(dim=1, keepdim=True)
        self.Ps[torch.arange(self.num_envs), s] /= sum_Ps_s
        self.Vs[torch.arange(self.num_envs), s] = valids
        self.Es[torch.arange(self.num_envs), s] = self.game.get_game_ended(canonical_boards, 1)
        return -v.squeeze(-1)

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