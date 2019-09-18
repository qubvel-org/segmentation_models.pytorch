import re
import torch.nn as nn


class Named(nn.Module):

    def __init__(self, name=None):
        self.__name__ = name or self._name

    @property
    def _name(self):
        name = self.__class__.__name__
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
