from torch_geometric.nn import fps, knn, radius
import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import PointConv
from torch_geometric.nn import global_max_pool


class SetAbstraction(torch.nn.Module):
    def __init__(self, ratio, radius, nsample, c_in, mlp):
        super().__init__()
        self.sampling_ratio = ratio
        self.radius = radius
        self.nsample = nsample
        self.conv = PointConv(set_pointnet_layers(c_in, mlp), add_self_loops=False)

    def forward(self, h, pos, batch):
        idx = fps(pos, batch, ratio=self.sampling_ratio)
        row, col = radius(x= pos, y= pos[idx], r= self.radius, batch_x=batch, batch_y=batch[idx], max_num_neighbors=self.nsample)
        edge_index = torch.stack([col, row], dim=0)
        #edge_index = knn(pos, pos[index], k=16, batch_x=batch, batch_y=batch[index])
        h_dst = None if h is None else h[idx]
        h = self.conv((h, h_dst), (pos, pos[idx]), edge_index)

        pos_new, batch_new = pos[idx], batch[idx]
        return h, pos_new, batch_new


class GlobalSetAbstraction(torch.nn.Module):
    def __init__(self, c_in, mlp):
        super().__init__()
        self.mlp = set_pointnet_layers(c_in, mlp)

    def forward(self, h, pos, batch):
        h = torch.cat([h, pos], dim=1)
        h = self.mlp(h)
        return global_max_pool(h, batch)

def set_pointnet_layers( in_channels, out_channels):
    return Sequential(Linear(in_channels, out_channels[0]),
                      BatchNorm1d(out_channels[0]),
                      ReLU(),
                      Linear(out_channels[0], out_channels[1]),
                      BatchNorm1d(out_channels[1]),
                      ReLU(),
                      Linear(out_channels[1], out_channels[2]),
                      BatchNorm1d(out_channels[2]),
                      ReLU())