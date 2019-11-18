import re
import torch.nn as nn

from pretrainedmodels.models.xception import pretrained_settings
from pretrainedmodels.models.xception import Xception


class XceptionEncoder(Xception):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        #modify padding to maintain output shape
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        
        self.pretrained = False
        del self.fc
        
        self.initialize()

    @staticmethod
    def _transition(x, transition_block):
        for module in transition_block:
            x = module(x)
            if isinstance(module, nn.ReLU):
                skip = x
        return x, skip

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x0 = self.relu(x)

        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x = self.block3(x2)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x3 = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x4 = self.bn4(x)
        features = [x4, x3, x2, x1, x0]
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
        'out_shapes': (2048, 728, 256, 128, 64),
        'params': {
        }
    },
}

