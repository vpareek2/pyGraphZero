import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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

class ChessResNet(nn.Module):
    def __init__(self, game, args):
        super(ChessResNet, self).__init__()
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        # Input layers
        self.conv = nn.Conv2d(12, args.num_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(args.num_channels)

        # Residual layers
        self.res_layers = nn.ModuleList([ResBlock(args.num_channels) for _ in range(args.num_residual_layers)])

        # Policy head
        self.policy_conv = nn.Conv2d(args.num_channels, 32, kernel_size=3, padding=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * self.board_x * self.board_y, self.action_size)

        # Value head
        self.value_conv = nn.Conv2d(args.num_channels, 3, kernel_size=3, padding=1)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * self.board_x * self.board_y, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, s):
        s = s.view(-1, 12, self.board_x, self.board_y)  # batch_size x 12 x 8 x 8
        s = F.relu(self.bn(self.conv(s)))

        for res_layer in self.res_layers:
            s = res_layer(s)

        # Policy head
        pi = F.relu(self.policy_bn(self.policy_conv(s)))
        pi = pi.view(-1, 32 * self.board_x * self.board_y)
        pi = self.policy_fc(pi)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(s)))
        v = v.view(-1, 3 * self.board_x * self.board_y)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return F.log_softmax(pi, dim=1), v

class NNetWrapper:
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

        self.nnet = ChessResNet(game, args).to(self.device)

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
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

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