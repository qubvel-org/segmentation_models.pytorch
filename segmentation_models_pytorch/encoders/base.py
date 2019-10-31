from typing import List


class EncoderMixin:
    @property
    def out_channels(self) -> List:
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]
