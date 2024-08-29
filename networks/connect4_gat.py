import os
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
import wandb

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


class Connect4GAT(nn.Module):
    def __init__(self, game, args):
        super(Connect4GAT, self).__init__()
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        self.num_nodes = self.board_x * self.board_y
        self.num_features = 3  # empty, player 1, player 2

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
        
        # Create edge index for Connect 4 board
        edge_index = self._create_connect4_edge_index(batch_size)
        
        return x, edge_index

    def _create_connect4_edge_index(self, batch_size):
        device = next(self.parameters()).device
        edge_list = []
        
        # Horizontal connections
        for row in range(self.board_y):
            for col in range(self.board_x - 1):
                node1 = row * self.board_x + col
                node2 = row * self.board_x + col + 1
                edge_list.extend([(node1, node2), (node2, node1)])
        
        # Vertical connections
        for row in range(self.board_y - 1):
            for col in range(self.board_x):
                node1 = row * self.board_x + col
                node2 = (row + 1) * self.board_x + col
                edge_list.extend([(node1, node2), (node2, node1)])
        
        # Diagonal connections (top-left to bottom-right)
        for row in range(self.board_y - 1):
            for col in range(self.board_x - 1):
                node1 = row * self.board_x + col
                node2 = (row + 1) * self.board_x + col + 1
                edge_list.extend([(node1, node2), (node2, node1)])
        
        # Diagonal connections (top-right to bottom-left)
        for row in range(self.board_y - 1):
            for col in range(1, self.board_x):
                node1 = row * self.board_x + col
                node2 = (row + 1) * self.board_x + col - 1
                edge_list.extend([(node1, node2), (node2, node1)])
        
        edge_index_single = torch.tensor(edge_list, dtype=torch.long, device=device).t()
        
        # Repeat the edge index for each graph in the batch
        edge_index = edge_index_single.repeat(1, batch_size)
        batch_offset = torch.arange(batch_size, device=device).repeat_interleave(edge_index_single.size(1)) * self.num_nodes
        edge_index = edge_index + batch_offset
        
        return edge_index

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

        self.nnet = Connect4GAT(game, args).to(self.device)

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

        # if self.is_main_process():
        #     wandb.init(project="tictactoe-gat", config=vars(args))
        #     wandb.watch(self.nnet)

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
            
            total_loss, pi_losses, v_losses = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_examples)
            
            if self.is_main_process():
                print(f'Epoch {epoch+1}/{self.args.epochs}')
                print(f'Validation Loss: {val_loss:.3f}')
                
                wandb.log({
                    "epoch": epoch,
                    "train_loss": total_loss / len(train_loader),
                    "train_pi_loss": pi_losses / len(train_loader),
                    "train_v_loss": v_losses / len(train_loader),
                    "val_loss": val_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
                
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
        total_loss = 0
        pi_losses = 0
        v_losses = 0
        
        for batch_idx, (boards, target_pis, target_vs) in enumerate(train_loader):
            boards, target_pis, target_vs = boards.to(self.device), target_pis.to(self.device), target_vs.to(self.device)
            
            # Apply data augmentation
            augmented_boards, augmented_pis = self.augment_batch(boards, target_pis)
            augmented_vs = target_vs.repeat(6)  # Repeat the target values for each augmentation
            
            self.optimizer.zero_grad()
            
            with autocast(device_type=self.device.type):
                out_pi, out_v = self.nnet(augmented_boards)
                l_pi = self.criterion_pi(out_pi, augmented_pis)
                l_v = self.criterion_v(out_v.squeeze(-1), augmented_vs)
                loss = l_pi + l_v

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pi_losses += l_pi.item()
            v_losses += l_v.item()

            if batch_idx % 100 == 0 and self.is_main_process():
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.3f}, Pi Loss: {l_pi.item():.3f}, V Loss: {l_v.item():.3f}')
                
                wandb.log({
                    "batch": batch_idx + epoch * len(train_loader),
                    "batch_loss": loss.item(),
                    "batch_pi_loss": l_pi.item(),
                    "batch_v_loss": l_v.item()
                })

        if self.args.distributed:
            total_loss = self.reduce_tensor(torch.tensor(total_loss).to(self.device))
            pi_losses = self.reduce_tensor(torch.tensor(pi_losses).to(self.device))
            v_losses = self.reduce_tensor(torch.tensor(v_losses).to(self.device))

        return total_loss, pi_losses, v_losses

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
        if board.shape != (self.board_y, self.board_x):
            raise ValueError(f"Invalid board shape. Expected ({self.board_y}, {self.board_x}), got {board.shape}")
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

        if self.is_main_process():
            wandb.save(filepath)

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
            
            # Horizontal flip
            flipped_board = np.fliplr(board)
            flipped_pi = np.zeros_like(pi)
            flipped_pi[:-1] = np.flip(pi[:-1])
            flipped_pi[-1] = pi[-1]  # Keep the pass move probability unchanged
            augmented.append((flipped_board, flipped_pi, v))
            
            # Random noise (slight perturbation)
            noisy_board = board + np.random.normal(0, 0.01, board.shape)
            noisy_board = np.clip(noisy_board, -1, 1)
            augmented.append((noisy_board, pi, v))
            
        return augmented
    
    def augment_batch(self, boards, pis):
        augmented_boards = []
        augmented_pis = []
        
        for board, pi in zip(boards, pis):
            board_np = board.squeeze().cpu().numpy()
            pi_np = pi.cpu().numpy()
            
            # Original
            augmented_boards.append(board_np)
            augmented_pis.append(pi_np)
            
            # Horizontal flip
            flipped_board = np.fliplr(board_np)
            flipped_pi = self.flip_policy(pi_np)
            augmented_boards.append(flipped_board)
            augmented_pis.append(flipped_pi)
        
        return (torch.FloatTensor(np.array(augmented_boards)).unsqueeze(1).to(self.device),
                torch.FloatTensor(np.array(augmented_pis)).to(self.device))

    def flip_policy(self, pi):
        flipped_pi = np.zeros_like(pi)
        flipped_pi[:-1] = np.flip(pi[:-1])
        flipped_pi[-1] = pi[-1]  # Keep the pass move probability unchanged
        return flipped_pi

    def is_main_process(self):
        return not self.args.distributed or (self.args.distributed and self.rank == 0)

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt