from __future__ import annotations
from typing import TYPE_CHECKING
from utils import read_yaml, upsample
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

    def forward(self, inputs: Tensor):
        return self.act(self.bn(self.conv(inputs)))

class ConvLayerSet(nn.Layer):
    def __init__(self, in_channels=1024):
        super().__init__()
        self.convs =  nn.Sequential(
            BasicConvLayer(in_channels, in_channels // 2, kernel_size=1, padding=0),
            BasicConvLayer(in_channels // 2, in_channels, kernel_size=3, padding=1),
            BasicConvLayer(in_channels, in_channels // 2, kernel_size=1, padding=0),
            BasicConvLayer(in_channels // 2, in_channels, kernel_size=3, padding=1),
            BasicConvLayer(in_channels, in_channels // 2, kernel_size=1, padding=0)
        )

    def forward(self, inputs: Tensor):
        return self.convs(inputs)


class ResidualBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cbl1 = BasicConvLayer(in_channels, out_channels, kernel_size=1, padding=0)
        self.cbl2 = BasicConvLayer(out_channels, out_channels * 2)

    def forward(self, inputs: Tensor):
        cbl1 = self.cbl1(inputs)
        cbl2 = self.cbl2(cbl1)
        return paddle.add(inputs, cbl2)


class ResidualXBlock(nn.Layer):
    def __init__(self, nums, in_channels):
        super().__init__()
        self.downSample = BasicConvLayer(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1)
        self.residuals = []
        for i in range(nums):
            residual = self.add_sublayer(f'residual{nums}_{i+1}', ResidualBlock(in_channels * 2, in_channels))
            self.residuals.append(residual)

    def forward(self, inputs: Tensor):
        x = self.downSample(inputs)
        for val in self.residuals:
            x = val(x)
        return x


class Yolo3_Backbone(nn.Layer):
    def __init__(self, in_channels=3):
        super().__init__()
        self.network_cfg = [1, 2, 8, 8, 4]
        self.conv0 = BasicConvLayer(in_channels, 32)
        self.res0 = ResidualXBlock(self.network_cfg[0], 32)
        self.res1= ResidualXBlock(self.network_cfg[1], 64)
        self.res2 = ResidualXBlock(self.network_cfg[2], 128)
        self.res3 = ResidualXBlock(self.network_cfg[3], 256)
        self.res4 = ResidualXBlock(self.network_cfg[4], 512)

    def forward(self, inputs):
        x = self.conv0(inputs)
        x = self.res0(x)
        x = self.res1(x)
        out0 = self.res2(x)  # res8
        out1 = self.res3(out0)  # res8
        out2 = self.res4(out1)  # res4
        return (out0, out1, out2)


class Yolo3_FPN(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv_set1 = ConvLayerSet()
        self.conv_set2 = ConvLayerSet(768)
        self.conv_set3 = ConvLayerSet(384)

        self.up_sample = upsample
        self.conv1 = BasicConvLayer(512, 256, kernel_size=1, padding=0)
        self.conv2 = BasicConvLayer(384, 128, kernel_size=1, padding=0)

    def forward(self, inputs):
        c2 = self.conv_set1(inputs[2])  # 13, 13, 1024
        x = self.conv1(c2)
        x = self.up_sample(x, mode='nearest')
        x = paddle.concat([x, inputs[1]], axis=1)  # 256+512=768
        c1 = self.conv_set2(x)  # 26, 26, 384
        x = self.conv2(c1)
        x = self.up_sample(x, mode='nearest')
        x = paddle.concat([x, inputs[0]], axis=1)  # 128+256=384
        c0 = self.conv_set3(x)  # 56, 56, 192
        return (c0, c1, c2)


class YOLOv3_Head(nn.Layer):
    def __init__(self, in_channels=512, num_classes=MODEL_CONFIG['MODEL']['NUM_CLASSES']):
        super().__init__()
        self.head1 = nn.Sequential(
            BasicConvLayer(in_channels, 1024),
            nn.Conv2D(1024, 3 * (5 + num_classes), 1)
        )
        self.head2 = nn.Sequential(
            BasicConvLayer(384, 768),
            nn.Conv2D(768, 3 * (5 + num_classes), 1)
        )
        self.head3 = nn.Sequential(
            BasicConvLayer(192, 384),
            nn.Conv2D(384, 3 *(5 + num_classes), 1)
        )

    def forward(self, inputs):
        out_s = self.head1(inputs[2])
        out_m = self.head2(inputs[1])
        out_l = self.head3(inputs[0])
        return (out_s, out_m, out_l)


class YOLOv3(nn.Layer):
    def __init__(self, in_channels=3):
        super().__init__()
        self.backbone = Yolo3_Backbone(in_channels)
        self.fpn = Yolo3_FPN()
        self.head = YOLOv3_Head()


    def forward(self, inputs):
        x = self.backbone(inputs)
        x = self.fpn(x)
        x = self.head(x)
        return x
