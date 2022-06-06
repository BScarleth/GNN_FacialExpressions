import torch
import torch.nn.functional as F
from torch.nn import Sequential, Dropout, Linear
from torch_geometric.nn import DynamicEdgeConv


class DGCNN(torch.nn.Module):
    def __init__(self, out_channels, k=30, aggr='max'):
        super().__init__()
        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.lin1 = MLP([3 * 64, 1024])
        self.mlp = Sequential(MLP([1024, 256]), Dropout(0.5),
        MLP([256, 128]), Dropout(0.5), Linear(128, out_channels))

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out = self.mlp(out)
        return F.normalize(out, p=2, dim=-1)