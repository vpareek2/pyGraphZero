import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.activation = activation
        self.dropout_prob = dropout_prob
        self.add_skip_connection = add_skip_connection
        
        self.linear_proj = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_heads, out_features))
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_heads, out_features))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_heads * out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        if add_skip_connection:
            self.skip_proj = nn.Linear(in_features, num_heads * out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)
        
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, node_features, edge_index):
        num_nodes = node_features.size(0)

        # Linear projection and regularization
        node_features = self.dropout(node_features)
        nodes_features_proj = self.linear_proj(node_features).view(-1, self.num_heads, self.out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)

        # Edge attention calculation
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)
        
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[1], num_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)

        # Neighborhood aggregation
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, node_features, num_nodes)

        # Skip connection and bias
        out_nodes_features = self.skip_concat_bias(node_features, out_nodes_features)

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[0]
        trg_nodes_index = edge_index[1]
        scores_source = scores_source.index_select(0, src_nodes_index)
        scores_target = scores_target.index_select(0, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(0, src_nodes_index)
        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)
        size = list(exp_scores_per_edge.shape)
        size[0] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        neighborhood_sums.scatter_add_(0, trg_index_broadcasted, exp_scores_per_edge)
        return neighborhood_sums.index_select(0, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[0] = num_of_nodes
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[1], nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(0, trg_index_broadcasted, nodes_features_proj_lifted_weighted)
        return out_nodes_features

    def skip_concat_bias(self, in_nodes_features, out_nodes_features):
        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_heads, self.out_features)

        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_heads * self.out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=1)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features

    def explicit_broadcast(self, this, other):
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        return this.expand_as(other)

class ChessGAT(nn.Module):
    def __init__(self, game, args):
        super(ChessGAT, self).__init__()
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        self.num_nodes = self.board_x * self.board_y
        self.num_features = 12  # 6 piece types * 2 colors

        # GAT layers
        self.gat1 = GATLayer(self.num_features, args.num_channels, num_heads=4, dropout_prob=0.3)
        self.gat2 = GATLayer(args.num_channels * 4, args.num_channels, num_heads=4, dropout_prob=0.3)
        
        # Policy head
        self.policy_conv = nn.Conv2d(args.num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * self.board_x * self.board_y, self.action_size)

        # Value head
        self.value_conv = nn.Conv2d(args.num_channels, 3, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * self.board_x * self.board_y, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, s):
        x, edge_index = self._board_to_graph(s)
        
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # Reshape for 2D convolutions
        x = x.view(-1, self.args.num_channels, self.board_x, self.board_y)

        # Policy head
        pi = F.relu(self.policy_bn(self.policy_conv(x)))
        pi = pi.view(-1, 32 * self.board_x * self.board_y)
        pi = self.policy_fc(pi)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 3 * self.board_x * self.board_y)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return F.log_softmax(pi, dim=1), v

    def _board_to_graph(self, s):
        batch_size = s.size(0)
        x = s.view(batch_size, 12, -1).transpose(1, 2).reshape(-1, 12)
        
        # Create edge index for Chess board
        edge_index = self._create_chess_edge_index(self.board_x, self.board_y)
        edge_index = edge_index.repeat(1, batch_size)
        batch_offset = torch.arange(batch_size, device=s.device).repeat_interleave(edge_index.size(1)) * self.num_nodes
        edge_index = edge_index + batch_offset.unsqueeze(0)
        
        return x, edge_index

    def _create_chess_edge_index(self, rows, cols):
        edges = []
        # Horizontal and vertical connections
        for r in range(rows):
            for c in range(cols):
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        edges.append((r * cols + c, nr * cols + nc))
        
        # Knight moves
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                        (1, -2), (1, 2), (2, -1), (2, 1)]
        for r in range(rows):
            for c in range(cols):
                for dr, dc in knight_moves:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        edges.append((r * cols + c, nr * cols + nc))
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index

class ChessNNetWrapper:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

        if args.distributed:
            self.device = torch.device(f"cuda:{args.local_rank}")
            torch.cuda.set_device(self.device)
            dist.init_process_group(backend='nccl')
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.nnet = ChessGAT(game, args).to(self.device)

        if args.distributed:
            self.nnet = DDP(self.nnet, device_ids=[args.local_rank])
        elif torch.cuda.device_count() > 1:
            self.nnet = nn.DataParallel(self.nnet)

        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def train(self, examples):
        for epoch in range(self.args.epochs):
            if self.args.distributed:
                self.train_sampler.set_epoch(epoch)
            
            print(f'Epoch {epoch+1}/{self.args.epochs}')
            self.nnet.train()
            total_loss, pi_losses, v_losses = 0, 0, 0
            batch_count = int(len(examples) / self.args.batch_size)
            
            for _ in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                batch_examples = [examples[i] for i in sample_ids]
                boards, target_pis, target_vs = list(zip(*batch_examples))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64)).to(self.device)
                target_pis = torch.FloatTensor(np.array(target_pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(target_vs)).to(self.device)

                # Compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v.squeeze(-1))
                total_loss = l_pi + l_v

                # Compute gradient and do SGD step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                total_loss += total_loss.item()
                pi_losses += l_pi.item()
                v_losses += l_v.item()

            if self.args.distributed:
                dist.all_reduce(total_loss)
                dist.all_reduce(pi_losses)
                dist.all_reduce(v_losses)
                total_loss /= dist.get_world_size()
                pi_losses /= dist.get_world_size()
                v_losses /= dist.get_world_size()

            print(f'Average Loss: {total_loss/batch_count:.3f} | '
                  f'Pi Loss: {pi_losses/batch_count:.3f} | '
                  f'V Loss: {v_losses/batch_count:.3f}')

    def predict(self, board):
        start = time.time()
        board = torch.FloatTensor(board.astype(np.float64)).unsqueeze(0).to(self.device)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        print(f'PREDICTION TIME TAKEN: {time.time() - start:.3f}')
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.makedirs(folder)
        else:
            print("Checkpoint Directory exists!")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"No model in path '{filepath}'")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.args.local_rank} if self.args.distributed else self.device
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])