from typing import Callable, Mapping, Union

from torch import Tensor

ActivationType = Union[str, Callable[..., Tensor], None]
AuxParamsType = Mapping[str, Union[int, str, float, None]]
