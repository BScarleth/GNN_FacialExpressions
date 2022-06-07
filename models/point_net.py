import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Dropout, BatchNorm1d
from torch_geometric.nn import global_max_pool
from .set_abstraction import SetAbstraction, GlobalSetAbstraction
from models.point_net_layer import  PointNetLayer

class PointNet(torch.nn.Module):
    def __init__(self, c_out, dp_rate=0.4):
        super().__init__()

        self.conv1 = SetAbstraction(ratio = 0.5 , radius= 0.2, nsample=32, c_in= 3, mlp = [64, 64, 128]) #64, 64, 128
        self.conv2 = SetAbstraction(ratio = 0.25 , radius= 0.4, nsample=64, c_in= 128 + 3, mlp = [128, 128, 256]) #128
        self.conv3 = GlobalSetAbstraction(c_in = 256 + 3, mlp = [256, 512, 1024]) #512
        #self.conv3 = PointNetLayer(c_in=256, mlp=[256, 512, 1024])

        self.fc1 = Sequential(Linear(1024, 512), BatchNorm1d(512), ReLU(), Dropout(dp_rate))
        self.fc2 = Sequential(Linear(512, 256), BatchNorm1d(256), ReLU(), Dropout(dp_rate))
        self.classifier = Linear(256, c_out)

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        x, pos, batch = self.conv1(h=x, pos=pos, batch=batch)
        x, pos, batch = self.conv2(h=x, pos=pos, batch=batch)
        x = self.conv3(h=x, pos=pos, batch=batch)

        fc = self.fc1(x)
        fc = self.fc2(fc)
        fc = self.classifier(fc)
        return fc
        #return F.log_softmax(fc, dim=1)

    def test(self, model_path, device):
        self.load_state_dict(torch.load(model_path, map_location=device))



