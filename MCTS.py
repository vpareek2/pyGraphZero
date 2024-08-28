import torch
import numpy as np

class MCTS:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tensors for parallel environments
        self.num_envs = args.num_parallel_envs
        self.action_size = game.get_action_size()

        # Tree structure
        self.max_nodes = args.max_nodes
        self.Qsa = torch.zeros((self.num_envs, self.max_nodes, self.action_size), device=self.device)
        self.Nsa = torch.zeros((self.num_envs, self.max_nodes, self.action_size), dtype=torch.long, device=self.device)
        self.Ns = torch.zeros((self.num_envs, self.max_nodes), dtype=torch.long, device=self.device)
        self.Ps = torch.zeros((self.num_envs, self.max_nodes, self.action_size), device=self.device)
        
        self.Es = torch.zeros((self.num_envs, self.max_nodes), device=self.device)
        self.Vs = torch.zeros((self.num_envs, self.max_nodes, self.action_size), dtype=torch.bool, device=self.device)

        self.next_node = torch.ones(self.num_envs, dtype=torch.long, device=self.device)

    def get_action_prob(self, canonical_boards, temp=1):
        for _ in range(self.args.num_mcts_sims):
            self.search(canonical_boards)

        s = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        counts = self.Nsa[torch.arange(self.num_envs), s]

        if temp == 0:
            best_actions = counts.argmax(dim=1)
            probs = torch.zeros_like(counts)
            probs.scatter_(1, best_actions.unsqueeze(1), 1)
        else:
            counts = counts.float()
            if temp != 1:
                counts = counts ** (1. / temp)
            counts = torch.clamp(counts, min=1e-8)  # Avoid division by zero
            probs = counts / counts.sum(dim=1, keepdim=True)
        
        # Ensure probabilities are non-negative and sum to 1
        probs = torch.clamp(probs, min=0)
        probs = probs / probs.sum(dim=1, keepdim=True)

        return probs

    def search(self, canonical_boards):
        env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        s = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Add a depth counter to limit recursion
        depth = 0
        max_depth = 1000  # Adjust this value as needed
        
        while env_mask.any() and depth < max_depth:
            ended = self.Es[torch.arange(self.num_envs), s] != 0
            if ended.any():
                env_mask[ended] = False
                continue

            unvisited = self.Ns[torch.arange(self.num_envs), s] == 0
            if unvisited.any():
                unvisited_mask = env_mask & unvisited
                v = self._expand(canonical_boards[unvisited_mask], s[unvisited_mask])
                self._backpropagate(s[unvisited_mask], torch.zeros_like(s[unvisited_mask]), v)
                env_mask[unvisited_mask] = False
                continue

            valid_moves = self.Vs[torch.arange(self.num_envs), s]
            uct_scores = self._uct_scores(s)
            uct_scores[~valid_moves] = float('-inf')
            a = uct_scores.argmax(dim=1)

            next_s, next_player = self.game.get_next_state(canonical_boards, 1, a)
            next_player = torch.full((self.num_envs,), next_player, device=self.device)
            canonical_boards = self.game.get_canonical_form(next_s, next_player)
            
            # Instead of recursive call, update s and continue the loop
            s = self.next_node[torch.arange(self.num_envs)]
            self.next_node += 1
            depth += 1

        # Handle case where max depth is reached
        if depth == max_depth:
            print(f"Warning: Max depth {max_depth} reached in MCTS search")

        return -self.Es[torch.arange(self.num_envs), s]

    def _expand(self, canonical_boards, s):
        pi, v = self.nnet.predict(canonical_boards)
        valids = self.game.get_valid_moves(canonical_boards, 1)
        
        # Ensure pi and valids have the same shape as self.Ps
        pi = pi.view(len(s), -1)
        valids = valids.view(len(s), -1)
        
        # Ensure that pi and valids have the correct number of actions
        if pi.shape[1] != self.action_size or valids.shape[1] != self.action_size:
            pi = torch.nn.functional.pad(pi, (0, self.action_size - pi.shape[1]))
            valids = torch.nn.functional.pad(valids, (0, self.action_size - valids.shape[1]))
        
        self.Ps[torch.arange(len(s)), s] = pi.to(self.device)
        self.Ps[torch.arange(len(s)), s] *= valids.to(self.device)
        sum_Ps_s = self.Ps[torch.arange(len(s)), s].sum(dim=1, keepdim=True)
        self.Ps[torch.arange(len(s)), s] /= sum_Ps_s
        self.Vs[torch.arange(len(s)), s] = valids.to(self.device).bool()  # Convert to boolean
        self.Es[torch.arange(len(s)), s] = self.game.get_game_ended(canonical_boards, 1)
        v = v.squeeze(-1)
        return -v

    def _uct_scores(self, s):
        Ns_sqrt = torch.sqrt(self.Ns[torch.arange(self.num_envs), s].unsqueeze(1))
        Qsa = self.Qsa[torch.arange(self.num_envs), s]
        Psa = self.Ps[torch.arange(self.num_envs), s]
        Nsa = self.Nsa[torch.arange(self.num_envs), s]
        
        uct = Qsa + self.args.cpuct * Psa * Ns_sqrt / (1 + Nsa)
        return uct

    def _backpropagate(self, s, a, v):
        env_indices = torch.arange(self.num_envs)
        self.Qsa[env_indices, s, a] = (self.Nsa[env_indices, s, a] * self.Qsa[env_indices, s, a] + v) / (self.Nsa[env_indices, s, a] + 1)
        self.Nsa[env_indices, s, a] += 1
        self.Ns[env_indices, s] += 1