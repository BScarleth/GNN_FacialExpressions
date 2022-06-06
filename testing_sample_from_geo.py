import os.path as osp
import os
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
import random
#from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius
from dataset.coma_local_dataset import CoMA
from dataset.modelnet_local_dataset import ModelNet
from trainer.utils import plot_original_face, increase_dataset
from torch_geometric.transforms import SamplePoints


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        print(nn)
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, 10], dropout=0.5, batch_norm=False)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x).log_softmax(dim=-1)


def train(epoch):
    model.train()

    loss_acc = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss_acc += float(loss)
        loss.backward()
        optimizer.step()
    return loss_acc/len(train_loader)

def cambio(lista):
    labels = {9: 1, 1: 0, 10: 2}
    lb = []
    for l in lista:
        lb.append(labels[int(l)])
    return torch.tensor(lb).to(device)

def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    path = 'data/ModelNet10'
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    #cwd = os.getcwd()
    #root='{}/../dataset_pruebas/'.format(cwd)
    dataset = ModelNet(path, transform=SamplePoints(num=1024), pre_transform=T.NormalizeScale())
    #dataset = CoMA(root="/home/brenda/Documents/master/thesis/IAS_gutierrez_2022/dataset/")#, transform=SamplePoints(num=1024))

    #train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    #test_dataset = ModelNet(path, '10', False, transform, pre_transform)
    #new_dataset = increase_dataset(dataset)
    #random.shuffle(new_dataset)
    #print(len(new_dataset))
    #dataset.shuffle()
    print(dataset)

    indx = int((len(dataset) * 80) / 100)
    train_dataset = dataset[:indx]
    test_dataset = dataset[indx:]

    print(len(train_dataset))
    print(len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=6, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training started")
    for epoch in range(1, 200):
        loss_acc_m = train(epoch)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Test: {test_acc:.4f}, Loss: {loss_acc_m:.4f}')