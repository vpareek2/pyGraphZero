import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

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
        batch_size = s.size(0)
        x = torch.zeros(batch_size * self.num_nodes, 3, device=s.device)
        x[:, 0] = (s == 0).float().view(-1)
        x[:, 1] = (s == 1).float().view(-1)
        x[:, 2] = (s == -1).float().view(-1)
        
        # Create fully connected edge index
        edge_index = torch.combinations(torch.arange(self.num_nodes), r=2).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index = edge_index.repeat(1, batch_size)
        batch_offset = torch.arange(batch_size, device=s.device).repeat_interleave(edge_index.size(1)) * self.num_nodes
        edge_index = edge_index + batch_offset.unsqueeze(0)
        
        return x, edge_index

class NNetWrapper:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

        if args.distributed:
            self.setup_distributed()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.nnet = TicTacToeGAT(game, args).to(self.device)

        if args.distributed:
            self.nnet = DDP(self.nnet, device_ids=[self.args.local_rank], output_device=self.args.local_rank)
        elif torch.cuda.device_count() > 1:
            self.nnet = nn.DataParallel(self.nnet)

        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr, weight_decay=args.l2_regularization)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        self.scaler = GradScaler()
        self.criterion_pi = nn.CrossEntropyLoss()
        self.criterion_v = nn.MSELoss()

        self.best_val_loss = float('inf')
        self.patience = 10
        self.wait = 0

    def setup_distributed(self):
        self.args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.args.local_rank}")
        torch.cuda.set_device(self.device)
        dist.init_process_group(backend='nccl')
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def train(self, examples):
        augmented_examples = self.augment_examples(examples)
        train_examples, val_examples = train_test_split(augmented_examples, test_size=0.2)

        train_data = TensorDataset(
            torch.FloatTensor([ex[0] for ex in train_examples]),
            torch.FloatTensor([ex[1] for ex in train_examples]),
            torch.FloatTensor([ex[2] for ex in train_examples])
        )
        
        if self.args.distributed:
            train_sampler = DistributedSampler(train_data, num_replicas=self.world_size, rank=self.rank)
            train_loader = DataLoader(train_data, batch_size=self.args.batch_size, sampler=train_sampler)
        else:
            train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True)

        for epoch in range(self.args.epochs):
            if self.args.distributed:
                train_sampler.set_epoch(epoch)
            
            self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_examples)
            
            if self.is_main_process():
                print(f'Epoch {epoch+1}/{self.args.epochs}')
                print(f'Validation Loss: {val_loss:.3f}')
                
                self.scheduler.step(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.wait = 0
                    self.save_checkpoint()
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        print("Early stopping")
                        break

            if self.args.distributed:
                dist.barrier()

    def train_epoch(self, train_loader, epoch):
        self.nnet.train()
        total_loss, pi_losses, v_losses = 0, 0, 0
        
        for batch_idx, (boards, target_pis, target_vs) in enumerate(train_loader):
            boards, target_pis, target_vs = boards.to(self.device), target_pis.to(self.device), target_vs.to(self.device)
            
            boards, target_pis = self.augment_batch(boards, target_pis)

            with autocast():
                out_pi, out_v = self.nnet(boards)
                l_pi = self.criterion_pi(out_pi, target_pis)
                l_v = self.criterion_v(out_v.squeeze(-1), target_vs)
                loss = l_pi + l_v

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pi_losses += l_pi.item()
            v_losses += l_v.item()

            if batch_idx % 100 == 0 and self.is_main_process():
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.3f}, Pi Loss: {l_pi.item():.3f}, V Loss: {l_v.item():.3f}')

        if self.args.distributed:
            total_loss = self.reduce_tensor(torch.tensor(total_loss).to(self.device))
            pi_losses = self.reduce_tensor(torch.tensor(pi_losses).to(self.device))
            v_losses = self.reduce_tensor(torch.tensor(v_losses).to(self.device))

        if self.is_main_process():
            print(f'Epoch {epoch+1} Average Loss: {total_loss/len(train_loader):.3f} | '
                  f'Pi Loss: {pi_losses/len(train_loader):.3f} | '
                  f'V Loss: {v_losses/len(train_loader):.3f}')

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

        if self.args.distributed:
            val_loss = self.reduce_tensor(torch.tensor(val_loss).to(self.device))

        return val_loss / len(val_examples)

    def predict(self, board):
        """
        board: np array with board
        """
        # Input validation
        if not isinstance(board, np.ndarray):
            raise ValueError(f"Invalid input type. Expected numpy array, got {type(board)}")
        if board.shape != (3, 3):
            raise ValueError(f"Invalid board shape. Expected (3, 3), got {board.shape}")
        if not np.issubdtype(board.dtype, np.number):
            raise ValueError(f"Invalid board data type. Expected numeric type, got {board.dtype}")
        if not np.all(np.isin(board, [-1, 0, 1])):
            raise ValueError("Invalid board values. Expected only -1, 0, or 1")

        # Prepare input
        board = torch.FloatTensor(board.astype(np.float64))
        board = board.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Move the input tensor to the same device as the model
        board = board.to(self.device)
        
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # Ensure pi has shape (1, 10)
        pi = pi.unsqueeze(0) if pi.dim() == 1 else pi
        v = v.unsqueeze(0) if v.dim() == 1 else v

        # Move tensors to CPU and detach from computation graph
        return pi.cpu().detach(), v.cpu().detach()

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        if not self.is_main_process():
            return

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save({
            'state_dict': self.nnet.module.state_dict() if isinstance(self.nnet, (nn.DataParallel, DDP)) else self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'wait': self.wait,
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"No model in path '{filepath}'")

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.args.local_rank} if self.args.distributed else self.device
        checkpoint = torch.load(filepath, map_location=map_location)

        if isinstance(self.nnet, (nn.DataParallel, DDP)):
            self.nnet.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.nnet.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.wait = checkpoint['wait']

    def augment_examples(self, examples):
        augmented = []
        for board, pi, v in examples:
            # Original example
            augmented.append((board, pi, v))
            
            # Rotations
            for k in range(1, 4):
                rotated_board = np.rot90(board, k)
                rotated_pi = np.rot90(pi.reshape(3, 3), k).flatten()
                augmented.append((rotated_board, rotated_pi, v))
            
            # Horizontal flip
            flipped_board = np.fliplr(board)
            flipped_pi = np.fliplr(pi.reshape(3, 3)).flatten()
            augmented.append((flipped_board, flipped_pi, v))
            
            # Vertical flip
            flipped_board = np.flipud(board)
            flipped_pi = np.flipud(pi.reshape(3, 3)).flatten()
            augmented.append((flipped_board, flipped_pi, v))
            
            # Diagonal flip (transpose)
            transposed_board = np.transpose(board)
            transposed_pi = np.transpose(pi.reshape(3, 3)).flatten()
            augmented.append((transposed_board, transposed_pi, v))
            
            # Random noise (slight perturbation)
            noisy_board = board + np.random.normal(0, 0.01, board.shape)
            noisy_board = np.clip(noisy_board, -1, 1)
            augmented.append((noisy_board, pi, v))
            
        return augmented

    def augment_batch(self, boards, pis):
        augmented_boards = []
        augmented_pis = []
        
        for board, pi in zip(boards, pis):
            board_np = board.cpu().numpy()
            pi_np = pi.cpu().numpy().reshape(3, 3)
            
            # Random rotation
            k = np.random.randint(0, 4)
            if k > 0:
                board_np = np.rot90(board_np, k)
                pi_np = np.rot90(pi_np, k)
            
            # Random flip
            if np.random.random() < 0.5:
                if np.random.random() < 0.5:
                    board_np = np.fliplr(board_np)
                    pi_np = np.fliplr(pi_np)
                else:
                    board_np = np.flipud(board_np)
                    pi_np = np.flipud(pi_np)
            
            # Random transpose
            if np.random.random() < 0.5:
                board_np = np.transpose(board_np)
                pi_np = np.transpose(pi_np)
            
            # Random noise (slight perturbation)
            if np.random.random() < 0.2:
                board_np = board_np + np.random.normal(0, 0.01, board_np.shape)
                board_np = np.clip(board_np, -1, 1)
            
            augmented_boards.append(board_np)
            augmented_pis.append(pi_np.flatten())
        
        return (torch.FloatTensor(np.array(augmented_boards)).to(self.device),
                torch.FloatTensor(np.array(augmented_pis)).to(self.device))

    def is_main_process(self):
        return not self.args.distributed or (self.args.distributed and self.rank == 0)

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt