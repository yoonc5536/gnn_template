import networkx as nx
import dgl
import torch
from src.nn.MLP import MultiLayerPerceptron as MLP
from src.nn.RelationalMPNN import RelationalMessagePassingNeuralNetwork as RelMPNN


def get_graph():
    g = nx.petersen_graph()
    g = dgl.DGLGraph(g)

    nf_dim, ef_dim, u_dim = 3, 7, 5
    nn, ne = g.number_of_nodes(), g.number_of_edges()

    nf, ef = torch.randn(nn, nf_dim), torch.randn(ne, ef_dim)
    ntpye = torch.randint(0, 2, size=(nn,))
    etpye = torch.randint(0, 2, size=(ne,))
    u = torch.randn(1, u_dim)
    g.ndata['h'] = nf
    g.ndata['type'] = ntpye
    g.edata['h'] = ef
    g.edata['type'] = etpye
    return g, u


if __name__ == "__main__":
    node_dim = 3
    edge_dim = 7
    edge_model_input_dim = node_dim * 2 + edge_dim
    edge_model_output_dim = 5

    node_model_input_dim = node_dim + edge_model_output_dim
    node_model_output_dim = 11

    edge_model_dict = {0: MLP(edge_model_input_dim, edge_model_output_dim),
                       1: MLP(edge_model_input_dim, edge_model_output_dim),
                       2: MLP(edge_model_input_dim, edge_model_output_dim),
                       3: MLP(edge_model_input_dim, edge_model_output_dim)}

    node_model_dict = {0: MLP(node_model_input_dim, node_model_output_dim),
                       1: MLP(node_model_input_dim, node_model_output_dim),
                       2: MLP(node_model_input_dim, node_model_output_dim),
                       3: MLP(node_model_input_dim, node_model_output_dim)}

    rmpnn = RelMPNN(edge_model_dict, node_model_dict)

    g, _ = get_graph()
    nf = g.ndata.pop('h')
    ef = g.edata.pop('h')
    unf, uef = rmpnn(g, nf, ef)
