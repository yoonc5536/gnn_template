from functools import partial

import dgl
import torch
import torch.nn as nn

from src.utils.graph_nn_utils import get_aggregator


class MessagePassingNeuralNetwork(nn.Module):
    """
    The base MPNN class that performs MPNN update from 'Relational inductive biases, deep learning, and graph networks' p.16
    For supporting various edge, and node model, we designed each model will be given from outside.

    """
    def __init__(self,
                 edge_model: nn.Module,
                 node_model: nn.Module,
                 node_aggregator: str = 'sum'):
        super(MessagePassingNeuralNetwork, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model

        # define message func
        self.message_func = dgl.function.copy_e('h', 'm')

        # set node aggregator
        self.node_aggr = get_aggregator(node_aggregator)

    def forward(self, g, nf, ef):
        """
        :param g: dgl.DGLGraph or dgl.BatchedDGLGraph
        :param nf: torch.tensor [#. nodes x node feature dim]
        :param ef: torch.tensor [#. edges x edge feature dim]
        :return: updated node, edge feature
        """

        g.ndata['h'] = nf
        g.edata['h'] = ef

        # perform edge update
        g.apply_edges(func=self.edge_update)
        # perform node update
        g.pull(g.nodes(), message_func=self.message_func, reduce_func=self.node_aggr)
        g.apply_nodes(func=self.node_update)

        updated_ef = g.edata.pop('h')
        updated_nf = g.ndata.pop('h')
        _ = g.ndata.pop('agg_m')
        return updated_nf, updated_ef

    def edge_update(self, edges):
        sender_nf = edges.src['h']
        receiver_nf = edges.dst['h']
        ef = edges.data['h']
        em_input = torch.cat([ef, sender_nf, receiver_nf], dim=-1)
        updated_ef = self.edge_model(em_input)
        return {'h': updated_ef}

    def node_update(self, nodes):
        agg_m = nodes.data['agg_m']
        nf = nodes.data['h']
        nm_input = torch.cat([agg_m, nf], dim=-1)
        updated_nf = self.node_model(nm_input)
        return {'h': updated_nf}
