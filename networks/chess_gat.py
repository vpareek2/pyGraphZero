import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import numpy as np
import os

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.activation = activation
        self.add_skip_connection = add_skip_connection

        self.linear_proj = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_heads, out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_heads, out_features))

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

    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        # Linear projection and regularization
        x = self.dropout(x)
        x = self.linear_proj(x).view(num_nodes, self.num_heads, self.out_features)
        x = self.dropout(x)

        # Edge attention calculation
        scores_source = (x * self.scoring_fn_source).sum(dim=-1)
        scores_target = (x * self.scoring_fn_target).sum(dim=-1)
        scores_source_lifted, scores_target_lifted, x_lifted = self.lift(scores_source, scores_target, x, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[1], num_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)

        # Neighborhood aggregation
        x_lifted_weighted = x_lifted * attentions_per_edge
        out_nodes_features = self.aggregate_neighbors(x_lifted_weighted, edge_index, num_nodes)

        # Skip connection and bias
        out_nodes_features = self.skip_concat_bias(x, out_nodes_features)

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

    def lift(self, scores_source, scores_target, x, edge_index):
        src_nodes_index = edge_index[0]
        trg_nodes_index = edge_index[1]
        scores_source = scores_source.index_select(0, src_nodes_index)
        scores_target = scores_target.index_select(0, trg_nodes_index)
        x_lifted = x.index_select(0, src_nodes_index)
        return scores_source, scores_target, x_lifted

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

    def aggregate_neighbors(self, x_lifted_weighted, edge_index, num_of_nodes):
        trg_index_broadcasted = self.explicit_broadcast(edge_index[1], x_lifted_weighted)
        size = list(x_lifted_weighted.shape)
        size[0] = num_of_nodes
        out_nodes_features = torch.zeros(size, dtype=x_lifted_weighted.dtype, device=x_lifted_weighted.device)
        out_nodes_features.scatter_add_(0, trg_index_broadcasted, x_lifted_weighted)
        return out_nodes_features

    def skip_concat_bias(self, x, out_nodes_features):
        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == x.shape[-1]:
                out_nodes_features += x.view(*out_nodes_features.shape)
            else:
                out_nodes_features += self.skip_proj(x).view(*out_nodes_features.shape)

        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_heads * self.out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=-2)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features

    def explicit_broadcast(self, this, other):
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        return this.expand_as(other)

class ChessGAT(nn.Module):
    def __init__(self, game, args):
        super().__init__()
        self.num_gat_layers = 4  # Deeper network

        # Initial embedding
        self.piece_embedding = nn.Linear(12, args.num_channels)

        # Multiple GAT layers
        self.gat_layers = nn.ModuleList([
            GATLayer(
                args.num_channels if i == 0 else args.num_channels * args.num_heads,
                args.num_channels,
                args.num_heads
            ) for i in range(self.num_gat_layers)
        ])

        # Layer norms after each GAT
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(args.num_channels * args.num_heads)
            for _ in range(self.num_gat_layers)
        ])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(args.num_channels * args.num_heads * self.num_nodes, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(512, self.action_size)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(args.num_channels * args.num_heads * self.num_nodes, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, s):
        x, edge_index, edge_attr = self._board_to_graph(s)
        x = self.piece_embedding(x)

        # GAT layers with residual connections
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            residual = x
            x = gat(x, edge_index, edge_attr)
            x = norm(x)
            if x.shape == residual.shape:  # Add residual if shapes match
                x = x + residual

        x = x.view(s.size(0), -1)

        # Policy and value heads
        pi = self.policy_head(x)
        v = self.value_head(x)

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def _board_to_graph(self, s):
        # Ensure s is a 4D tensor (batch_size, 12, 8, 8)
        if s.dim() == 3:
            s = s.unsqueeze(0)

        batch_size, channels, height, width = s.shape
        assert channels == 12 and height == 8 and width == 8, "Input should be (batch_size, 12, 8, 8)"

        # Reshape s to (batch_size * num_nodes, 12)
        x = s.view(batch_size * self.num_nodes, 12)

        # Create edge index for Chess graph structure
        edge_index_single = []
        for i in range(self.board_x):
            for j in range(self.board_y):
                node = i * self.board_y + j
                # Horizontal and vertical connections
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.board_x and 0 <= nj < self.board_y:
                        neighbor = ni * self.board_y + nj
                        edge_index_single.extend([[node, neighbor], [neighbor, node]])
                # Diagonal connections
                for di, dj in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.board_x and 0 <= nj < self.board_y:
                        neighbor = ni * self.board_y + nj
                        edge_index_single.extend([[node, neighbor], [neighbor, node]])

        edge_index_single = torch.tensor(edge_index_single, device=s.device).t()

        # Repeat the edge index for each graph in the batch
        edge_index = edge_index_single.repeat(1, batch_size)
        batch_offset = torch.arange(batch_size, device=s.device).repeat_interleave(edge_index_single.size(1)) * self.num_nodes
        edge_index = edge_index + batch_offset

        return x, edge_index

class NNetWrapper:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.nnet = ChessGAT(game, args).to(self.device)

        if args.distributed:
            self.nnet = DDP(self.nnet, device_ids=[args.local_rank], output_device=args.local_rank)
        elif torch.cuda.device_count() > 1:
            self.nnet = nn.DataParallel(self.nnet)

        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr, weight_decay=args.l2_regularization)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        self.scaler = GradScaler()
        self.criterion_pi = nn.CrossEntropyLoss()
        self.criterion_v = nn.MSELoss()

    def train(self, examples):
        train_examples, val_examples = train_test_split(examples, test_size=0.2)

        train_data = TensorDataset(
            torch.FloatTensor([ex[0] for ex in train_examples]),
            torch.FloatTensor([ex[1] for ex in train_examples]),
            torch.FloatTensor([ex[2] for ex in train_examples])
        )

        train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True)

        for epoch in range(self.args.epochs):
            self.nnet.train()
            total_loss = 0
            for batch_idx, (boards, target_pis, target_vs) in enumerate(train_loader):
                boards, target_pis, target_vs = boards.to(self.device), target_pis.to(self.device), target_vs.to(self.device)

                self.optimizer.zero_grad()

                with autocast(device_type=self.device.type):
                    out_pi, out_v = self.nnet(boards)
                    l_pi = self.criterion_pi(out_pi, target_pis)
                    l_v = self.criterion_v(out_v.squeeze(-1), target_vs)
                    loss = l_pi + l_v

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()

            val_loss = self.validate(val_examples)
            self.scheduler.step(val_loss)

    def validate(self, val_examples):
        self.nnet.eval()
        val_loss = 0
        with torch.no_grad():
            for board, target_pi, target_v in val_examples:
                board = torch.FloatTensor(board.astype(np.float64)).unsqueeze(0).to(self.device)
                target_pi = torch.FloatTensor(target_pi).unsqueeze(0).to(self.device)
                target_v = torch.FloatTensor([target_v]).to(self.device)

                out_pi, out_v = self.nnet(board)
                l_pi = self.criterion_pi(out_pi, target_pi)
                l_v = self.criterion_v(out_v.squeeze(-1), target_v)
                val_loss += (l_pi + l_v).item()

        return val_loss / len(val_examples)

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float64))
        if board.dim() == 3:
            board = board.unsqueeze(0)
        board = board.to(self.device)

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return pi.exp().cpu().numpy()[0], v.cpu().numpy()[0].item()


    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
        }, filepath, _use_new_zipfile_serialization=True)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"No model in path '{filepath}'")

        # Use weights_only=True to avoid potential security issues
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)

        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])

    def augment_examples(self, examples):
        augmented = []
        for board, pi, v in examples:
            # Original example
            augmented.append((board, pi, v))

            # Horizontal flip
            flipped_board = np.fliplr(board)
            flipped_pi = np.flip(pi)
            augmented.append((flipped_board, flipped_pi, v))

        return augmented
