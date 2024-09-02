import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x = self.linear_proj(x).view(-1, self.num_heads, self.out_features)
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
                out_nodes_features += x.unsqueeze(1)
            else:
                out_nodes_features += self.skip_proj(x).view(-1, self.num_heads, self.out_features)

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

class TicTacToeGAT(nn.Module):
    def __init__(self, game, args):
        super(TicTacToeGAT, self).__init__()
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        self.num_nodes = self.board_x * self.board_y
        self.num_features = 3  # empty, X, O

        self.gat1 = GATLayer(self.num_features, args.num_channels, num_heads=4, dropout_prob=0.3)
        self.gat2 = GATLayer(args.num_channels * 4, args.num_channels, num_heads=4, dropout_prob=0.3)
        
        self.fc1 = nn.Linear(args.num_channels * 4 * self.num_nodes, 256)
        self.fc2 = nn.Linear(256, 128)
        
        self.fc_policy = nn.Linear(128, self.action_size)
        self.fc_value = nn.Linear(128, 1)

    def forward(self, s):
        x, edge_index = self._board_to_graph(s)
        
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        x = x.view(-1, self.args.num_channels * 4 * self.num_nodes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        pi = self.fc_policy(x)
        v = self.fc_value(x)

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def _board_to_graph(self, s):
        # Ensure s is a 4D tensor (batch_size, channels, height, width)
        if s.dim() == 3:
            s = s.unsqueeze(1)
        elif s.dim() == 2:
            s = s.unsqueeze(0).unsqueeze(0)
        
        batch_size, channels, height, width = s.shape
        
        # Reshape s to (batch_size * num_nodes, channels)
        s_flat = s.view(batch_size, channels, -1).transpose(1, 2).contiguous().view(-1, channels)
        
        x = torch.zeros(batch_size * self.num_nodes, 3, device=s.device)
        x[:, 0] = (s_flat == 0).float().sum(dim=1)
        x[:, 1] = (s_flat == 1).float().sum(dim=1)
        x[:, 2] = (s_flat == -1).float().sum(dim=1)
        
        # Create fully connected edge index for a single graph
        edge_index_single = torch.combinations(torch.arange(self.num_nodes, device=s.device), r=2).t()
        edge_index_single = torch.cat([edge_index_single, edge_index_single.flip(0)], dim=1)
        
        # Repeat the edge index for each graph in the batch
        edge_index = edge_index_single.repeat(1, batch_size)
        batch_offset = torch.arange(batch_size, device=s.device).repeat_interleave(edge_index_single.size(1)) * self.num_nodes
        edge_index = edge_index + batch_offset
        
        return x, edge_index