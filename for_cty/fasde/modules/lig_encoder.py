import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear
from torch_geometric.nn import MessagePassing, radius_graph
from math import pi as PI

from .lig_encoder_utils import flatten_graph, unflatten_graph


class GaussianSmearing(Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift



class CFConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_filters, edge_channels, cutoff=10.0):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = Sequential(
            Linear(edge_channels, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )   # Network for generating filter weights
        self.cutoff = cutoff
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.nn[0].weight)
        self.nn[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.nn[2].weight)
        self.nn[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_length, edge_attr):
        W = self.nn(edge_attr)

        if self.cutoff is not None:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)     # Modification: cutoff
            W = W * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(Module):

    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, num_gaussians, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNetEncoder(Module):

    def __init__(self, h_channels=202, e_channel=5, hidden_channels=128, num_filters=128,
                num_interactions=6, edge_channels=64, cutoff=10.0):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.node_act = Linear(h_channels, hidden_channels, bias=False)
        self.edge_act = Linear(e_channel, edge_channels, bias=False)
        self.cutoff = cutoff

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff)
            self.interactions.append(block)
        self.reset_parameters()

    def reset_parameters(self):
        for interaction in self.interactions:
            interaction.reset_parameters()

    @property
    def out_channels(self):
        return self.hidden_channels

    def forward(self, node_attr, pos, edge_index, edge_attr):
        batch_size= pos.shape[0]

        node_attr, edge_attr, edge_index = flatten_graph(
            node_attr, edge_attr, edge_index)

        flatten_pos = torch.flatten(pos, 0, 1)
        edge_length = torch.norm(flatten_pos[edge_index[0]] - flatten_pos[edge_index[1]] + 1e-10, dim=1)
        
        edge_attr_ = self.distance_expansion(edge_length)

        node_attr = self.node_act(node_attr)
        edge_attr = self.edge_act(edge_attr) + edge_attr_

        for interaction in self.interactions:
            node_attr = node_attr + interaction(node_attr, edge_index, edge_length, edge_attr)

        node_attr = unflatten_graph(node_attr, batch_size)
        return node_attr


class LinearEncoder(Module):
    def __init__(self, h_channels, hidden_channels=128, ) -> None:
        super().__init__()

        self.node_feature = Linear(h_channels, hidden_channels, bias=False)

    def forward(self, node_attr, pos, edge_index, edge_attr):
        
        return self.node_feature(node_attr)


