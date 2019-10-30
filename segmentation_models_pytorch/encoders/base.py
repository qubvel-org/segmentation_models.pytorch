from typing import List


class EncoderMixin:

    @property
    def out_channels(self):
        """Return list of channel dimentions for decoder output features"""
        return self._out_channels[:self._depth + 1]
