import dgl
import torch
from torch.utils.data import Dataset, DataLoader


class DGLGraphDataset(Dataset):

    def __init__(self, x_graphs, y_graphs):
        assert isinstance(x_graphs, list), "expected spec of 'x_graphs' is a list of graphs"
        assert isinstance(y_graphs, list), "expected spec of 'y_graphs' is a list of graphs"
        assert len(x_graphs) == len(y_graphs)

        self.x_data = x_graphs
        self.y_data = y_graphs
        self.len = len(x_graphs)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.len


class DGLGraphDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        if 'device' in kwargs:
            device = kwargs.pop('device')
        else:
            device = 'cpu'

        self.device = device
        collate_fn = self.collate_fn
        kwargs['collate_fn'] = collate_fn
        super(DGLGraphDataLoader, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        x = dgl.batch([item[0] for item in batch])
        y = dgl.batch([item[1] for item in batch])

        x.to(torch.device(self.device))
        y.to(torch.device(self.device))
        return [x, y]


class DGLGraphTensorDataset(Dataset):

    def __init__(self, graphs, target):
        assert isinstance(graphs, list), "expected spec of 'x_graphs' is a list of graphs"

        assert len(graphs) == target.shape[0]

        self.x_data = graphs
        self.y_data = target
        self.len = len(graphs)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.len


class DGLGraphTensorDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        if 'device' in kwargs:
            device = kwargs.pop('device')
        else:
            device = 'cpu'

        self.device = device
        collate_fn = self.collate_fn
        kwargs['collate_fn'] = collate_fn
        super(DGLGraphTensorDataLoader, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        x = dgl.batch([item[0] for item in batch])
        y = torch.cat([item[1] for item in batch], dim=0)

        x.to(torch.device(self.device))
        y = y.to(self.device)
        return [x, y]
