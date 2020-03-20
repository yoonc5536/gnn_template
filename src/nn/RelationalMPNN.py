import dgl
import torch
import torch.nn as nn

from src.utils.graph_nn_utils import get_aggregator


class RelationalMessagePassingNeuralNetwork(nn.Module):
    """
    The base MPNN class that performs MPNN update from 'Relational inductive biases, deep learning, and graph networks' p.16
    For supporting various edge, and node model, we designed each model will be given from outside.

    """

    def __init__(self,
                 edge_model_dict: dict,
                 node_model_dict: dict,
                 node_aggregator: str = 'sum'):
        super(RelationalMessagePassingNeuralNetwork, self).__init__()
        self.edge_model_dict = edge_model_dict
        self.node_model_dict = node_model_dict
        self.edge_types = list(edge_model_dict.keys())
        self.node_types = list(node_model_dict.keys())

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
        etype = edges.data['type']
        em_input = torch.cat([ef, sender_nf, receiver_nf], dim=-1)

        num_edges = sender_nf.shape[0]
        type_pos_list = []
        updated_efs = []
        for type in self.edge_types:  # looping over list! to preserve the looping order
            type_pos = torch.arange(num_edges)[etype == type]
            if len(type_pos) == 0:
                continue
            enc = self.edge_model_dict[type]
            inp = em_input[type_pos, :]

            type_pos_list.append(type_pos)
            updated_efs.append(enc(inp))

        type_perm = torch.cat(type_pos_list, dim=0).view(-1).long()
        updated_efs = torch.cat(updated_efs, dim=0)
        updated_efs = updated_efs[type_perm, :]
        return {'h': updated_efs}

    def node_update(self, nodes):
        agg_m = nodes.data['agg_m']
        nf = nodes.data['h']
        ntype = nodes.data['type']
        nm_input = torch.cat([agg_m, nf], dim=-1)

        num_nodes = nf.shape[0]
        type_pos_list = []
        updated_nfs = []
        for type in self.node_types:  # looping over list! to preserve the looping order
            type_pos = torch.arange(num_nodes)[ntype == type]
            if len(type_pos) == 0:
                continue
            enc = self.node_model_dict[type]
            inp = nm_input[type_pos, :]

            type_pos_list.append(type_pos)
            updated_nfs.append(enc(inp))

        type_perm = torch.cat(type_pos_list, dim=0).view(-1).long()
        updated_nfs = torch.cat(updated_nfs, dim=0)
        updated_nfs = updated_nfs[type_perm, :]
        return {'h': updated_nfs}
