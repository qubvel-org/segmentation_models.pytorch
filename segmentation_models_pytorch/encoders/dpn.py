import numpy as np

import torch
import torch.nn.functional as F

from pretrainedmodels.models.dpn import DPN
from pretrainedmodels.models.dpn import pretrained_settings


class DPNEncorder(DPN):

    def __init__(self, feature_blocks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_blocks = np.cumsum(feature_blocks)
        self.pretrained = False
        
        del self.last_linear

    def forward(self, x):

        features = []

        input_block = self.features[0]

        x = input_block.conv(x)
        x = input_block.bn(x)
        x = input_block.act(x)
        features.append(x)

        x = input_block.pool(x)

        for i, module in enumerate(self.features[1:], 1):
            x = module(x)
            if i in self.feature_blocks:
                features.append(x)

        out_features = [
            features[4],
            F.relu(torch.cat(features[3], dim=1), inplace=True),
            F.relu(torch.cat(features[2], dim=1), inplace=True),
            F.relu(torch.cat(features[1], dim=1), inplace=True),
            features[0],
        ]

        return out_features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('last_linear.bias')
        state_dict.pop('last_linear.weight')
        super().load_state_dict(state_dict, **kwargs)


dpn_encoders = {
    'dpn68': {
        'encoder': DPNEncorder,
        'out_shapes': (832, 704, 320, 144, 10),
        'pretrained_settings': pretrained_settings['dpn68'],
        'params': {
            'feature_blocks': (3, 4, 12, 4),
            'groups': 32,
            'inc_sec': (16, 32, 32, 64),
            'k_r': 128,
            'k_sec': (3, 4, 12, 3),
            'num_classes': 1000,
            'num_init_features': 10,
            'small': True,
            'test_time_pool': True
        },
    },

    'dpn68b': {
        'encoder': DPNEncorder,
        'out_shapes': (832, 704, 320, 144, 10),
        'pretrained_settings': pretrained_settings['dpn68b'],
        'params': {
            'feature_blocks': (3, 4, 12, 4),
            'b': True,
            'groups': 32,
            'inc_sec': (16, 32, 32, 64),
            'k_r': 128,
            'k_sec': (3, 4, 12, 3),
            'num_classes': 1000,
            'num_init_features': 10,
            'small': True,
            'test_time_pool': True,
        },
    },

    'dpn92': {
        'encoder': DPNEncorder,
        'out_shapes': (2688, 1552, 704, 336, 64),
        'pretrained_settings': pretrained_settings['dpn92'],
        'params': {
            'feature_blocks': (3, 4, 20, 4),
            'groups': 32,
            'inc_sec': (16, 32, 24, 128),
            'k_r': 96,
            'k_sec': (3, 4, 20, 3),
            'num_classes': 1000,
            'num_init_features': 64,
            'test_time_pool': True
        },
    },

    'dpn98': {
        'encoder': DPNEncorder,
        'out_shapes': (2688, 1728, 768, 336, 96),
        'pretrained_settings': pretrained_settings['dpn98'],
        'params': {
            'feature_blocks': (3, 6, 20, 4),
            'groups': 40,
            'inc_sec': (16, 32, 32, 128),
            'k_r': 160,
            'k_sec': (3, 6, 20, 3),
            'num_classes': 1000,
            'num_init_features': 96,
            'test_time_pool': True,
        },
    },

    'dpn107': {
        'encoder': DPNEncorder,
        'out_shapes': (2688, 2432, 1152, 376, 128),
        'pretrained_settings': pretrained_settings['dpn107'],
        'params': {
            'feature_blocks': (4, 8, 20, 4),
            'groups': 50,
            'inc_sec': (20, 64, 64, 128),
            'k_r': 200,
            'k_sec': (4, 8, 20, 3),
            'num_classes': 1000,
            'num_init_features': 128,
            'test_time_pool': True
        },
    },

    'dpn131': {
        'encoder': DPNEncorder,
        'out_shapes': (2688, 1984, 832, 352, 128),
        'pretrained_settings': pretrained_settings['dpn131'],
        'params': {
            'feature_blocks': (4, 8, 28, 4),
            'groups': 40,
            'inc_sec': (16, 32, 32, 128),
            'k_r': 160,
            'k_sec': (4, 8, 28, 3),
            'num_classes': 1000,
            'num_init_features': 128,
            'test_time_pool': True
        },
    },

}
