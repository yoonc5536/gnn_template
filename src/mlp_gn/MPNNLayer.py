import torch.nn as nn

from src.nn.MLP import MultiLayerPerceptron as MLP
from src.nn.MPNN import MessagePassingNeuralNetwork as GNLayer


class MPNNLayer(nn.Module):

    def __init__(self,
                 node_in_dim: int,
                 edge_in_dim: int,
                 node_out_dim: int,
                 edge_out_dim: int,
                 node_aggregator: str = 'sum',
                 mlp_params: dict = {}):
        super(MPNNLayer, self).__init__()

        default_mlp_params = dict()  # common params for node, edge, global model
        default_mlp_params['num_neurons'] = [64, 64]
        default_mlp_params['activation'] = 'ReLU'
        default_mlp_params['out_activation'] = 'ReLU'
        mlp_params.update(default_mlp_params)

        em_input_dim = edge_in_dim + 2 * node_in_dim
        nm_input_dim = edge_out_dim + node_in_dim

        mlp_params.update({'input_dimension': em_input_dim, 'output_dimension': edge_out_dim})
        edge_mlp = MLP(**mlp_params)

        mlp_params.update({'input_dimension': nm_input_dim, 'output_dimension': node_out_dim})
        node_mlp = MLP(**mlp_params)

        self.gn = GNLayer(node_model=node_mlp,
                          edge_model=edge_mlp,
                          node_aggregator=node_aggregator)

    def forward(self, g, nf, ef):
        u_nf, u_ef = self.gn(g, nf, ef)
        return u_nf, u_ef
