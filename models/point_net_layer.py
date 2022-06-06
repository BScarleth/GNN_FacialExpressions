from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
import torch

class PointNetLayer(MessagePassing):
    def __init__(self, c_in, mlp):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(c_in + 3, mlp[0]),
                              ReLU(),
                              Linear(mlp[0], mlp[1]),
                              ReLU(),
                              Linear(mlp[1], mlp[2]),
                              ReLU())

    def forward(self, x, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=x, pos=pos)
    #self.conv(h=h, pos=pos, edge_index=edge_index)
    def message(self, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)

        return self.mlp(input)  # Apply our final MLP.