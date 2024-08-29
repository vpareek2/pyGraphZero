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
from torch.amp import GradScaler, autocast
import wandb

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
        out = F.relu(out)
        return out

class Connect4ResNet(nn.Module):
    def __init__(self, game, args):
        super(Connect4ResNet, self).__init__()
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
        s = s.view(-1, 1, self.board_x, self.board_y)
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

        return F.log_softmax(pi, dim=1), torch.tanh(v)

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

        self.nnet = Connect4ResNet(game, args).to(self.device)

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

        if self.is_main_process():
            wandb.init(project="connect4-resnet", config=vars(args))
            wandb.watch(self.nnet)

    def setup_distributed(self):
        self.args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.args.local_rank}")
        torch.cuda.set_device(self.device)
        dist.init_process_group(backend='nccl')
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def train(self, examples):
        if not examples:
            print("No examples to train on. Skipping training.")
            return

        augmented_examples = self.augment_examples(examples)
        train_examples, val_examples = train_test_split(augmented_examples, test_size=0.2)

        boards = np.array([ex[0] for ex in train_examples])
        pis = np.array([ex[1] for ex in train_examples])
        vs = np.array([ex[2] for ex in train_examples])

        train_data = TensorDataset(
            torch.FloatTensor(boards),
            torch.FloatTensor(pis),
            torch.FloatTensor(vs)
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
            
            self.optimizer.zero_grad()
            
            with autocast(device_type=self.device.type):
                out_pi, out_v = self.nnet(boards)
                loss_pi = self.criterion_pi(out_pi, target_pis)
                loss_v = self.criterion_v(out_v, target_vs)
                total_loss = loss_pi + loss_v
            
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += total_loss.item()
            pi_losses += loss_pi.item()
            v_losses += loss_v.item()
        
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
        board = torch.FloatTensor(board.astype(np.float64))
        board = board.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        board = board.to(self.device)
        
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        pi = pi.unsqueeze(0) if pi.dim() == 1 else pi
        v = v.unsqueeze(0) if v.dim() == 1 else v

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
            if board.ndim == 2:
                board = board[np.newaxis, :, :]
            
            augmented.append((board, pi, v))
            
            flipped_board = np.flip(board, axis=2)
            flipped_pi = self.flip_policy(pi)
            augmented.append((flipped_board, flipped_pi, v))
        
        return augmented

    def flip_policy(self, pi):
        board_policy = pi[:self.board_x].reshape(1, self.board_x)
        flipped_board_policy = np.fliplr(board_policy)
        return np.concatenate([flipped_board_policy.flatten(), pi[self.board_x:]])

    def is_main_process(self):
        return not self.args.distributed or (self.args.distributed and self.rank == 0)

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt