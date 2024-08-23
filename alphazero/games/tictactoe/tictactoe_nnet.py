import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TicTacToeNNet(nn.Module):
    def __init__(self, game, args):
        super(TicTacToeNNet, self).__init__()
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels * (self.board_x - 2) * (self.board_y - 2), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(-1, 1, self.board_x, self.board_y)  # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, self.args.num_channels * (self.board_x - 2) * (self.board_y - 2))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.softmax(pi, dim=1), torch.tanh(v)

    def compile(self, args):
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        self.criterion_pi = nn.CrossEntropyLoss()
        self.criterion_v = nn.MSELoss()

    def train_step(self, boards, target_pis, target_vs):
        # Perform forward pass
        out_pi, out_v = self(boards)

        # Compute losses
        l_pi = self.criterion_pi(out_pi, target_pis)
        l_v = self.criterion_v(out_v.view(-1), target_vs)
        total_loss = l_pi + l_v

        # Perform backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), l_pi.item(), l_v.item()