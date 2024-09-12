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
from torch.amp import GradScaler, autocast

class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class TicTacToeResNet(nn.Module):
    def __init__(self, game, args):
        super(TicTacToeResNet, self).__init__()
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        self.conv = nn.Conv2d(1, args.num_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(args.num_channels)
        self.res_blocks = nn.ModuleList([ResBlock(args.num_channels) for _ in range(args.num_res_blocks)])

        self.pi_conv = nn.Conv2d(args.num_channels, 32, kernel_size=1)
        self.pi_bn = nn.BatchNorm2d(32)
        self.pi_fc = nn.Linear(32 * self.board_x * self.board_y, self.action_size)

        self.v_conv = nn.Conv2d(args.num_channels, 32, kernel_size=1)
        self.v_bn = nn.BatchNorm2d(32)
        self.v_fc1 = nn.Linear(32 * self.board_x * self.board_y, 256)
        self.v_fc2 = nn.Linear(256, 1)

        self.dropout = nn.Dropout(p=args.dropout_rate)

    def forward(self, s):
        print(f"Input shape: {s.shape}")
        s = s.view(-1, 1, self.board_x, self.board_y)
        print(f"Reshaped input: {s.shape}")
        s = F.relu(self.bn(self.conv(s)))

        for res_block in self.res_blocks:
            s = res_block(s)

        pi = F.relu(self.pi_bn(self.pi_conv(s)))
        pi = pi.view(-1, 32 * self.board_x * self.board_y)
        pi = self.dropout(pi)
        pi = self.pi_fc(pi)

        v = F.relu(self.v_bn(self.v_conv(s)))
        v = v.view(-1, 32 * self.board_x * self.board_y)
        v = self.dropout(v)
        v = F.relu(self.v_fc1(v))
        v = self.dropout(v)
        v = self.v_fc2(v)

        return F.log_softmax(pi, dim=1), v.squeeze(-1)

class NNetWrapper:
    def __init__(self, game, args):
        self.args = args
        self.game = game
        self.device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        print(f"Initializing network on device: {self.device}")
        
        try:
            # Initialize the network
            self.nnet = TicTacToeResNet(game, args)
            
            # Move the network to the specified device
            self.nnet = self.nnet.to(self.device)
            print("Network successfully moved to device")
        except Exception as e:
            print(f"Error initializing network: {str(e)}")
            raise

    def setup_device(self):
        if self.args.distributed:
            self.args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{self.args.local_rank}")
            torch.cuda.set_device(device)
            dist.init_process_group(backend='nccl')
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def train(self, examples):
        if not examples:
            print("No examples to train on. Skipping training.")
            return

        train_data, val_data = self.prepare_data(examples)
        train_loader = self.get_data_loader(train_data)

        for epoch in range(self.args.epochs):
            if self.args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_data)

            if self.is_main_process():
                print(f'Epoch {epoch+1}/{self.args.epochs}')
                print(f'Train Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}')

                self.scheduler.step(val_loss)

            if self.args.distributed:
                dist.barrier()

    def train_epoch(self, train_loader):
        self.nnet.train()
        total_loss = 0.0

        for boards, target_pis, target_vs in train_loader:
            boards, target_pis, target_vs = self.to_device(boards, target_pis, target_vs)

            self.optimizer.zero_grad()

            with autocast('cuda'):
                out_pi, out_v = self.nnet(boards)
                loss_pi = self.criterion_pi(out_pi, target_pis)
                loss_v = self.criterion_v(out_v, target_vs.squeeze())
                loss = loss_pi + loss_v

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_data):
        self.nnet.eval()
        val_loss = 0
        val_loader = DataLoader(val_data, batch_size=self.args.batch_size, shuffle=False)

        with torch.no_grad():
            for boards, target_pis, target_vs in val_loader:
                boards, target_pis, target_vs = self.to_device(boards, target_pis, target_vs)
                out_pi, out_v = self.nnet(boards)
                loss_pi = self.criterion_pi(out_pi, target_pis)
                loss_v = self.criterion_v(out_v, target_vs.squeeze())
                val_loss += (loss_pi + loss_v).item()

        val_loss /= len(val_loader)

        if self.args.distributed:
            val_loss = self.reduce_tensor(torch.tensor(val_loss, device=self.device)).item()

        return val_loss

    def predict(self, board):
        board = self.preprocess_board(board)
        print(f"Board device: {board.device}")
        print(f"Network device: {next(self.nnet.parameters()).device}")
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        return pi, v  # Return PyTorch tensors directly

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
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"No model in path '{filepath}'")

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.args.local_rank} if self.args.distributed else self.device
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)

        if isinstance(self.nnet, (nn.DataParallel, DDP)):
            self.nnet.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.nnet.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])

    def prepare_data(self, examples):
        augmented_examples = self.augment_examples(examples)
        train_examples, val_examples = train_test_split(augmented_examples, test_size=0.2)

        train_data = self.examples_to_tensors(train_examples)
        val_data = self.examples_to_tensors(val_examples)

        return train_data, val_data

    def get_data_loader(self, data):
        if self.args.distributed:
            sampler = DistributedSampler(data, num_replicas=self.world_size, rank=self.rank)
            return DataLoader(data, batch_size=self.args.batch_size, sampler=sampler)
        else:
            return DataLoader(data, batch_size=self.args.batch_size, shuffle=True)

    def augment_examples(self, examples):
        augmented = []
        for board, pi, v in examples:
            augmented.extend(self.get_symmetries(board, pi, v))
        return augmented

    def get_symmetries(self, board, pi, v):
        symmetries = []
        for i in range(4):
            for flip in [False, True]:
                new_b = np.rot90(board, i)
                new_pi = np.rot90(pi.reshape(3, 3), i)
                if flip:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)
                symmetries.append((new_b, new_pi.flatten(), v))
        return symmetries

    def examples_to_tensors(self, examples):
        boards, pis, vs = zip(*examples)
        boards = torch.FloatTensor(np.array(boards)).unsqueeze(1)  # Shape: (N, 1, 3, 3)
        pis = torch.FloatTensor(np.array(pis))                     # Shape: (N, 9)
        vs = torch.FloatTensor(np.array(vs))                       # Shape: (N,)
        return TensorDataset(boards, pis, vs)

    def to_device(self, *tensors):
        return (t.to(self.device) for t in tensors)

    def preprocess_board(self, board):
        if isinstance(board, np.ndarray):
            board = torch.from_numpy(board).float()
        elif not isinstance(board, torch.Tensor):
            raise TypeError("Board must be either a NumPy array or a PyTorch tensor")
        
        board = board.to(self.device)
        
        if board.dim() == 2:
            board = board.unsqueeze(0).unsqueeze(0)
        elif board.dim() == 3:
            board = board.unsqueeze(0)
        
        return board

    def is_main_process(self):
        return not self.args.distributed or (self.args.distributed and self.rank == 0)

    def reduce_tensor(self, tensor):
        if not self.args.distributed:
            return tensor
        
        if not dist.is_initialized():
            return tensor

        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt