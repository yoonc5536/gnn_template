from functools import partial

import dgl
import torch
import torch.nn as nn

from src.utils.graph_nn_utils import get_aggregator, get_node_readout, get_edge_readout


class GraphNetwork(nn.Module):

    def __init__(self,
                 edge_model: nn.Module,
                 node_model: nn.Module,
                 global_model: nn.Module,
                 node_aggregator: str = 'sum',
                 global_aggregator: str = 'sum'):
        super(GraphNetwork, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        # define message func
        self.message_func = dgl.function.copy_e('h', 'm')

        # set node aggregator
        self.node_aggr = get_aggregator(node_aggregator)

        # set global node, edge aggregator
        self.global_edge_aggr = get_node_readout(global_aggregator)
        self.global_node_aggr = get_edge_readout(global_aggregator)

    def forward(self, g, nf, ef, u):
        """
        :param g: dgl.DGLGraph or dgl.BatchedDGLGraph
        :param nf: torch.tensor [#. nodes x node feature dim]
        :param ef: torch.tensor [#. edges x edge feature dim]
        :param u: torch.tensor [#. graphs x global feature dim]
        :return: updated node, edge, global feature
        """

        assert u.dim() == 2, "expected global feature spec : [#.num_graphs x global feat dim]"
        g.ndata['h'] = nf
        g.edata['h'] = ef

        # preparing global features
        if type(g) == dgl.DGLGraph:
            e_repeat = g.number_of_edges()
            n_repeat = g.number_of_nodes()
            num_g = 1
        elif type(g) == dgl.BatchedDGLGraph:
            e_repeat = torch.tensor(g.batch_num_edges)
            n_repeat = torch.tensor(g.batch_num_nodes)
            num_g = g.batch_size
        else:
            raise RuntimeError("graph must be either an instance of dgl.DGLGraph or dgl.BatchedDGLGraph")

        ef_u = u.repeat_interleave(e_repeat, dim=0)
        nf_u = u.repeat_interleave(n_repeat, dim=0)

        # perform edge update
        edge_update = partial(self.edge_update, u=ef_u)
        g.apply_edges(func=edge_update)

        # perform node update
        g.pull(g.nodes(), message_func=self.message_func, reduce_func=self.node_aggr)
        node_update = partial(self.node_update, u=nf_u)
        g.apply_nodes(func=node_update)

        # perform global feature update
        edge_aggr = self.global_edge_aggr(g)
        node_aggr = self.global_node_aggr(g)

        gm_input = torch.cat([edge_aggr.view(num_g, -1), node_aggr.view(num_g, -1), u], dim=-1)
        updated_u = self.global_model(gm_input)

        # get updated edge feature and node feature
        updated_ef = g.edata.pop('h')
        updated_nf = g.ndata.pop('h')
        _ = g.ndata.pop('agg_m')
        return updated_nf, updated_ef, updated_u

    def edge_update(self, edges, u):
        sender_nf = edges.src['h']
        receiver_nf = edges.dst['h']
        ef = edges.data['h']
        em_input = torch.cat([ef, sender_nf, receiver_nf, u], dim=-1)
        updated_ef = self.edge_model(em_input)
        return {'h': updated_ef}

    def node_update(self, nodes, u):
        agg_m = nodes.data['agg_m']
        nf = nodes.data['h']
        nm_input = torch.cat([agg_m, nf, u], dim=-1)
        updated_nf = self.node_model(nm_input)
        return {'h': updated_nf}


if __name__ == "__main__":
    import networkx as nx
    import dgl
    from src.nn.MLP import MultiLayerPerceptron as MLP

    nf_dim, ef_dim, u_dim = 3, 7, 5
    nh_dim = 5
    eh_dim = 7
    uh_dim = 11


    def get_graph():
        g = nx.petersen_graph()
        g = dgl.DGLGraph(g)

        nn, ne = g.number_of_nodes(), g.number_of_edges()

        nf, ef = torch.randn(nn, nf_dim), torch.randn(ne, ef_dim)
        u = torch.randn(1, u_dim)
        g.ndata['h'] = nf
        g.edata['h'] = ef
        return g, u


    em_input_dim = ef_dim + 2 * nf_dim + u_dim
    nm_input_dim = eh_dim + nf_dim + u_dim
    um_input_dim = eh_dim + nh_dim + u_dim

    node_model = MLP(input_dimension=nm_input_dim,
                     output_dimension=nh_dim)
    edge_model = MLP(input_dimension=em_input_dim,
                     output_dimension=eh_dim)
    u_model = MLP(input_dimension=um_input_dim,
                  output_dimension=uh_dim)

    gn = GraphNetwork(edge_model=edge_model,
                      node_model=node_model,
                      global_model=u_model)

    g1, u1 = get_graph()
    g2, u2 = get_graph()
    batched_g = dgl.batch([g1, g2])
    batched_u = torch.cat([u1, u2], dim=0)

    batched_nf = batched_g.ndata['h']
    batched_ef = batched_g.edata['h']

    bunf, buef, buu = gn(batched_g, batched_nf, batched_ef, batched_u)
