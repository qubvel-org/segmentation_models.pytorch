from typing import List

from ._base import EncoderMixin
from ._xception import Xception


class XceptionEncoder(Xception, EncoderMixin):
    def __init__(
        self,
        out_channels: List[int],
        *args,
        depth: int = 5,
        output_stride: int = 32,
        **kwargs,
    ):
        if depth > 5 or depth < 1:
            raise ValueError(
                f"{self.__class__.__name__} depth should be in range [1, 5], got {depth}"
            )
        super().__init__(*args, **kwargs)

        self._depth = depth
        self._in_channels = 3
        self._out_channels = out_channels
        self._output_stride = output_stride

        # modify padding to maintain output shape
        self.conv1.padding = (1, 1)
        self.conv2.padding = (1, 1)

        del self.fc

    def make_dilated(self, *args, **kwargs):
        raise ValueError(
            "Xception encoder does not support dilated mode "
            "due to pooling operation for downsampling!"
        )

    def forward(self, x):
        features = [x]

        if self._depth >= 1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            features.append(x)

        if self._depth >= 2:
            x = self.block1(x)
            features.append(x)

        if self._depth >= 3:
            x = self.block2(x)
            features.append(x)

        if self._depth >= 4:
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            features.append(x)

        if self._depth >= 5:
            x = self.block12(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.conv4(x)
            x = self.bn4(x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict):
        # remove linear
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)

        super().load_state_dict(state_dict)


xception_encoders = {
    "xception": {
        "encoder": XceptionEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/xception.imagenet",
                "revision": "01cfaf27c11353b1f0c578e7e26d2c000ea91049",
            },
        },
        "params": {"out_channels": [3, 64, 128, 256, 728, 2048]},
    }
}
