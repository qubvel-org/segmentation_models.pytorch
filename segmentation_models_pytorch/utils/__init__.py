import warnings

from . import train
from . import losses
from . import metrics

__all__ = ["train", "losses", "metrics"]

warnings.warn(
    "`smp.utils` module is deprecated and will be removed in future releases.",
    DeprecationWarning,
)
