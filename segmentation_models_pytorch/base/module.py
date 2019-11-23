import torch
from torch import nn


class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path='new_state_dict.pth'):
        torch.save(self.state_dict(), path)
        print('saving module in', path)

    def load(self, path=None):
        if not path:
            path = 'new_state_dict.pth'
        self.load_state_dict(torch.load(path))
        print('loading module from', path)
