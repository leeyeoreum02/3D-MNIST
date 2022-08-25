import torch
import torch.nn as nn


def mlpblock(in_channels, out_channels, act_f=True):
    layers = [
        nn.Conv1d(in_channels, out_channels, 1),
        nn.BatchNorm1d(out_channels),
    ]
    if act_f:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def fcblock(in_channels, out_channels, dropout_rate=None):
    layers = [
        nn.Linear(in_channels, out_channels),
    ]
    if dropout_rate is not None:
        layers.append(nn.Dropout(p=dropout_rate))
    layers += [
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    ]
    return nn.Sequential(*layers)

class TNet(nn.Module):
    def __init__(self, dim=64):
        super(TNet, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            mlpblock(dim, 64),
            mlpblock(64, 128),
            mlpblock(128, 1024)
        )
        self.fc = nn.Sequential(
            fcblock(1024, 512),
            fcblock(512, 256),
            nn.Linear(256, dim*dim)
        )
        
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.fc(x)

        idt = torch.eye(self.dim, dtype=torch.float32).flatten().unsqueeze(0).repeat(x.size()[0], 1)
        idt = idt.to(x.device)
        x = x + idt
        x = x.view(-1, self.dim, self.dim)
        return x


class PointNetCls(nn.Module):
    def __init__(self, num_classes=10):
        super(PointNetCls, self).__init__()

        self.tnet = TNet(dim=3)
        self.mlp1 = mlpblock(3, 64)

        self.tnet_feature = TNet(dim=64)

        self.mlp2 = nn.Sequential(
            mlpblock(64, 128),
            mlpblock(128, 1024, act_f=False)
        )

        self.mlp3 = nn.Sequential(
            fcblock(1024, 512),
            fcblock(512, 256, dropout_rate=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        :input size: (N, n_points, 3)
        :output size: (N, num_classes)
        """
        x = x.transpose(2, 1) #N, 3, n_points
        trans = self.tnet(x) #N, 3, 3
        x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1) #N, 3, n_points
        x = self.mlp1(x) #N, 64, n_points

        trans_feat = self.tnet_feature(x) #N, 64, 64
        x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1) #N, 64, n_points

        x = self.mlp2(x) #N, 1024, n_points
        x = torch.max(x, 2, keepdim=False)[0] #N, 1024 (global feature)

        x = self.mlp3(x) #N, num_classes

        return x, trans_feat