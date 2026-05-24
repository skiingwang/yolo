from __future__ import annotations

from typing import TYPE_CHECKING
from utils import read_yaml
import paddle, paddle.nn as nn

if TYPE_CHECKING:
    from paddle import Tensor
    from typing import Tuple

__all__ = []


MODEL_CONFIG = read_yaml('./config.yaml')

class BasicConvLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2D(
            num_features=out_channels,
            weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(0., 0.02),
                                         regularizer=paddle.regularizer.L2Decay(0.))
        )
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Darknet19(nn.Layer):
    def __init__(self, in_channels=3, num_classes=MODEL_CONFIG['MODEL']['NUM_CLASSES']):
        super().__init__()
        self.num_classes = num_classes
        self.network_cfg = [
            (32, 3, 1, 1), 'M',
            (64, 3, 1, 1), 'M',
            (128, 3, 1, 1), (64, 1, 1, 0), (128, 3, 1, 1), 'M',
            (256, 3, 1, 1), (128, 1, 1, 0), (256, 3, 1, 1), 'M',
            [
                (512, 3, 1, 1), (256, 1, 1, 0), 2
            ],
            (512, 3, 1, 1), 'M',
            [
                (1024, 3, 1, 1), (512, 1, 1, 0), 2
            ],
            (1024, 3, 1, 1)
        ]
        self.features = self._build_features(self.network_cfg, in_channels)
        self.head = self._build_head()

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

    def _build_features(self, cfg, in_channels):
        backbone = []
        for i,c in enumerate(cfg):
            match c:
                case tuple():
                    backbone.append(self.add_sublayer(f'conv_{i+1}', BasicConvLayer(in_channels, c[0], c[1], c[2], c[3])))
                    in_channels = c[0]
                case 'M':
                    backbone.append(self.add_sublayer(f'maxpool_{i+1}', nn.MaxPool2D(2, 2)))
                case list():
                    for _ in range(c[2]):
                        backbone.append(self.add_sublayer(f'conv_{i+1}', BasicConvLayer(in_channels, c[0][0], c[0][1], c[0][2], c[0][3])))
                        backbone.append(self.add_sublayer(f'conv_{i+2}', BasicConvLayer(c[0][0], c[1][0], c[1][1], c[1][2], c[1][3])))
                        in_channels = c[1][0]
        return nn.Sequential(*backbone)

    def _build_head(self):
        return nn.Sequential(
            nn.Conv2D(1024, self.num_classes, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
