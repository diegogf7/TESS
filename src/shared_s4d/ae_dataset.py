import torch

from src.shared_s4d.dataset import AreaGroupLOODataset

class AreaGroupAEDataset(AreaGroupLOODataset):
    def __getitem__(self, idx):

        rows, _a = self.items[idx]
        return torch.tensor(self.X[rows]), torch.tensor(self.M[rows])
