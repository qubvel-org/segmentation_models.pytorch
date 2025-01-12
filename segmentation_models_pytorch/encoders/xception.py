from pretrainedmodels.models.xception import Xception

from ._base import EncoderMixin


class XceptionEncoder(Xception, EncoderMixin):
    def __init__(self, out_channels, *args, depth=5, **kwargs):
        super().__init__(*args, **kwargs)

        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

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
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
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
            x = self.relu(x)
            x = self.conv4(x)
            x = self.bn4(x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict):
        # remove linear
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)

        super().load_state_dict(state_dict)


pretrained_settings = {
    "xception": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth",
            "input_space": "RGB",
            "input_size": [3, 299, 299],
            "input_range": [0, 1],
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "num_classes": 1000,
            "scale": 0.8975,  # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}

xception_encoders = {
    "xception": {
        "encoder": XceptionEncoder,
        "pretrained_settings": pretrained_settings["xception"],
        "params": {"out_channels": (3, 64, 128, 256, 728, 2048)},
    }
}
