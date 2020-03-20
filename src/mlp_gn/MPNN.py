from copy import deepcopy as dc
import torch.nn as nn
from src.mlp_gn.MPNNLayer import MPNNLayer


class MPNN(nn.Module):
    """
    A stack of MPNN layers
    """

    def __init__(self,
                 node_in_dim: int,
                 edge_in_dim: int,
                 node_hidden_dim: int = 64,
                 edge_hidden_dim: int = 64,
                 node_out_dim: int = 64,
                 edge_out_dim: int = 64,
                 num_hidden_gn: int = 0,
                 node_aggregator: str = 'sum',
                 mlp_params: dict = {}):
        super(MPNN, self).__init__()
        node_in_dims = [node_in_dim] + num_hidden_gn * [node_hidden_dim]
        node_out_dims = num_hidden_gn * [node_hidden_dim] + [node_out_dim]

        edge_in_dims = [edge_in_dim] + num_hidden_gn * [edge_hidden_dim]
        edge_out_dims = num_hidden_gn * [edge_hidden_dim] + [edge_out_dim]

        self.layers = nn.ModuleList()
        for ni, no, ei, eo in zip(node_in_dims, node_out_dims,
                                  edge_in_dims, edge_out_dims):
            gn = MPNNLayer(node_in_dim=ni,
                           node_out_dim=no,
                           edge_in_dim=ei,
                           edge_out_dim=eo,
                           node_aggregator=node_aggregator,
                           mlp_params=dc(mlp_params))

            self.layers.append(gn)

    def forward(self, g, nf, ef):
        for gn in self.layers:
            nf, ef = gn(g, nf, ef)
        return nf, ef
