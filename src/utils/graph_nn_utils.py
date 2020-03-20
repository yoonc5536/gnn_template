import dgl

AGGR_TYPES = ['sum', 'mean']


def get_aggregator(mode, from_field='m', to_field='agg_m'):
    if mode in AGGR_TYPES:
        if mode == 'sum':
            aggr = dgl.function.sum(from_field, to_field)
        if mode == 'mean':
            aggr = dgl.function.mean(from_field, to_field)
    else:
        raise RuntimeError("Given aggregation mode {} is not supported".format(mode))
    return aggr


def get_node_readout(mode, from_field='h'):
    def sum_readout(g):
        return dgl.sum_nodes(g, from_field)

    def mean_readout(g):
        return dgl.mean_nodes(g, from_field)

    if mode in AGGR_TYPES:
        if mode == 'sum':
            read_func = sum_readout
        if mode == 'mean':
            read_func = mean_readout
    else:
        raise RuntimeError("Given readout mode {} is not supported".format(mode))
    return read_func


def get_edge_readout(mode, from_field='h'):
    def sum_readout(g):
        return dgl.sum_edges(g, from_field)

    def mean_readout(g):
        return dgl.mean_edges(g, from_field)

    if mode in AGGR_TYPES:
        if mode == 'sum':
            read_func = sum_readout
        if mode == 'mean':
            read_func = mean_readout
    else:
        raise RuntimeError("Given readout mode {} is not supported".format(mode))
    return read_func
