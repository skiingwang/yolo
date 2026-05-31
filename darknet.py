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

    def forward(self, inputs:Tensor):
        return self.act(self.bn(self.conv(inputs)))

class ResidualBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cbl1 = BasicConvLayer(in_channels, out_channels, kernel_size=1, padding=0)
        self.cbl2 = BasicConvLayer(out_channels, out_channels*2)

    def forward(self, inputs:Tensor):
        cbl1 = self.cbl1(inputs)
        cbl2 = self.cbl2(cbl1)
        return paddle.add(inputs, cbl2)

class ResidualXBlock(nn.Layer):
    def __init__(self, nums, in_channels):
        super().__init__()
        self.downSample = BasicConvLayer(in_channels, in_channels*2, kernel_size=3, stride=2, padding=1)
        self.residuals = []
        for i in range(nums):
            residual = self.add_sublayer(f'residual{nums}_{i+1}', ResidualBlock(in_channels*2, in_channels))
            self.residuals.append(residual)

    def forward(self, inputs:Tensor):
        x = self.downSample(inputs)
        for val in self.residuals:
            x = val(x)
        return x

class Darknet53(nn.Layer):
    def __init__(self, in_channels=3, num_classes=MODEL_CONFIG['MODEL']['NUM_CLASSES']):
        super().__init__()
        self.num_classes = num_classes
        self.network_cfg = [1, 2, 8, 8, 4]
        self.features = self._build_features(in_channels)
        self.head = self._build_head()

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

    def _build_features(self, in_channels):
        backbone = []
        downSample_inchannels = 32

        for i in self.network_cfg:
            backbone.append(ResidualXBlock(i, downSample_inchannels))
            downSample_inchannels *= 2
        return nn.Sequential(
            BasicConvLayer(in_channels, 32),
            *backbone
        )

    def _build_head(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=self.num_classes),
            nn.Softmax()
        )
