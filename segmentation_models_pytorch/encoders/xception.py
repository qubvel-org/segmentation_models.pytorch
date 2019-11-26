import re
import torch.nn as nn

from pretrainedmodels.models.xception import pretrained_settings
from pretrainedmodels.models.xception import Xception

from ._base import EncoderMixin

class XceptionEncoder(Xception, EncoderMixin):

    def __init__(self, out_channels, *args, depth=5, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3
        
        #modify padding to maintain output shape
        self.conv1.padding = 1
        self.conv2.padding = 1
        
        del self.fc

    @staticmethod
    def _transition(x, transition_block):
        for module in transition_block:
            x = module(x)
            if isinstance(module, nn.ReLU):
                skip = x
        return x, skip

    def forward(self, x):
        features = [x]
        
        if self._depth > 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x0 = self.relu(x)
            features.append(x0)
            
        if self._depth > 1:
            x1 = self.block1(x0)
            features.append(x1)
            
        if self._depth > 2:
            x2 = self.block2(x1)
            features.append(x2)
        
        if self._depth > 3:
            x = self.block3(x2)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
            x3 = self.block11(x)
            features.append(x3)
            
        if self._depth > 4:
            x = self.block12(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)

            x = self.conv4(x)
            x4 = self.bn4(x)
            features.append(x4)
            
        return features
       

    def load_state_dict(self, state_dict):
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        # remove linear
        state_dict.pop('fc.bias')
        state_dict.pop('fc.weight')

        super().load_state_dict(state_dict)


xception_encoders = {
    'xception': {
        'encoder': XceptionEncoder,
        'pretrained_settings': pretrained_settings['xception'],
        'params': {
            'out_channels': (3, 64, 128, 256, 728, 2048),
        }
    },
}

