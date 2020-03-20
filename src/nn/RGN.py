import torch
import torch.nn as nn
import dgl
from src.nn.GN import GraphNetwork


class RecurrentGraphNetwork(nn.Module):
    """
    Recurrent Graph Network from "Graph Networks as Learnable Physics Engines for Inference and Control"
    https://arxiv.org/pdf/1806.01242.pdf
    """

    def __init__(self,
                 edge_hidden_dim: int,
                 node_hidden_dim: int,
                 global_hidden_dim: int,
                 edge_model: nn.Module,
                 node_model: nn.Module,
                 global_model: nn.Module,
                 node_aggregator: str = 'sum',
                 global_aggregator: str = 'sum'):

        super(RecurrentGraphNetwork, self).__init__()
        self.edge_hidden_dim = edge_hidden_dim
        self.node_hidden_dim = node_hidden_dim
        self.global_hidden_dim = global_hidden_dim

        self.core_gn = GraphNetwork(edge_model=edge_model,
                                    node_model=node_model,
                                    global_model=global_model,
                                    node_aggregator=node_aggregator,
                                    global_aggregator=global_aggregator)

    def forward(self, g, nf, ef, u, hnf=None, hef=None, hu=None):
        device = nf.device
        if hnf is None:
            hnf, hef, hu = self.prepare_init_hidden(g, device)

        # graph-concat
        nf = torch.cat([hnf, nf], dim=-1)
        ef = torch.cat([hef, ef], dim=-1)
        u = torch.cat([hu, u], dim=-1)

        # gn update
        unf, uef, uu = self.core_gn(g, nf, ef, u)

        # graph-split
        hnf, unf = unf.split([self.node_hidden_dim, unf.shape[1] - self.node_hidden_dim], dim=-1)
        hef, uef = uef.split([self.edge_hidden_dim, uef.shape[1] - self.edge_hidden_dim], dim=-1)
        hu, uu = uu.split([self.global_hidden_dim, uu.shape[1] - self.global_hidden_dim], dim=-1)

        return (hnf, hef, hu), (unf, uef, uu)

    def prepare_init_hidden(self, g, device):
        ne = g.number_of_edges()
        nn = g.number_of_nodes()
        if type(g) == dgl.DGLGraph:
            num_g = 1
        elif type(g) == dgl.BatchedDGLGraph:
            num_g = g.batch_size

        hnf = torch.zeros(nn, self.node_hidden_dim).to(device)
        hef = torch.zeros(ne, self.node_hidden_dim).to(device)
        hu = torch.zeros(num_g, self.node_hidden_dim).to(device)

        return hnf, hef, hu
