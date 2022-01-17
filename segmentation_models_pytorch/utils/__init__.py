import warnings

from . import train
from . import losses
from . import metrics

warnings.warn(
    "`smp.utils` module is deprecated and will be removed in future releases.",
    DeprecationWarning,
)
