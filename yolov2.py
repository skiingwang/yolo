from __future__ import annotations

from typing import TYPE_CHECKING
from utils import read_yaml, passthrough
import paddle, paddle.nn as nn

if TYPE_CHECKING:
    from paddle import Tensor
    from typing import Tuple

__all__ = []


MODEL_CONFIG = read_yaml('./config.yaml')

class BasicConvLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2D(
            num_features=out_channels,
            weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(0., 0.02), regularizer=paddle.regularizer.L2Decay(0.))
        )
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

"""YOLOv2模型
训练数据集：PASCAL VOC 2012
输入图像：448x448x3
网格大小：7x7
Label：[x, y, w, h, objectness, cls1, cls2, ...]
x, y：目标物体的中心坐标相对于网格的坐标，范围为[0, 1]（即局部归一化坐标）。
w, h：目标物体的宽度和高度相对于画布的尺寸，范围为[0, 1]（即全局归一化尺寸）。
objectness：如果目标物体的中心位于网格中，为1，否则为0。
Outputs: [x, y, w, h, confidence, cls1, cls2, ...]
confidence：目标物体的置信度，Pr(obj)*IOU，范围为[0, 1]。
"""

class Yolo2(nn.Layer):
    def __init__(self, in_channels=3, split_size=MODEL_CONFIG['MODEL']['SPLIT_SIZE'], num_classes=MODEL_CONFIG['MODEL']['NUM_CLASSES'], num_boxes=MODEL_CONFIG['MODEL']['NUM_BOXES']):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.s = split_size  # 网格大小 13x13
        self.network_cfg1 = [
            (32, 3, 1, 1), 'M',  # 416 -> 208
            (64, 3, 1, 1), 'M',  # 208 -> 104
            (128, 3, 1, 1), (64, 1, 1, 0), (128, 3, 1, 1), 'M',  # 104 -> 52
            (256, 3, 1, 1), (128, 1, 1, 0), (256, 3, 1, 1), 'M',  # 52 -> 26
            [
                (512, 3, 1, 1), (256, 1, 1, 0), 2
            ]  # 26 -> 26
        ]
        self.network_cfg_passthrough = [
            (512, 3, 1, 1)
        ]
        self.network_cfg2 = [
            'M',
            [
                (1024, 3, 1, 1), (512, 1, 1, 0), 2
            ],
            (1024, 3, 1, 1)
        ]
        self.passthrouth_conv = nn.Conv2D(512, 64, 1, 1, 0)
        self.backbone_1 = self._build_backbone(self.network_cfg1, in_channels)
        self.backbone_passthrough = self._build_backbone(self.network_cfg_passthrough, 256)
        self.backbone_2 = self._build_backbone(self.network_cfg2, 512)
        self.head_1 = self._build_head()
        self.head_2 = nn.Sequential(
            BasicConvLayer(1280, 1024, 3, 1, 1),
            nn.Conv2D(1024, 5 * (5 + self.num_classes), 1, 1, 0),
        )

    def _build_backbone(self, cfg, in_channels):
        backbone = []
        for c in cfg:
            match c:
                case tuple():
                    backbone.append(BasicConvLayer(in_channels, c[0], c[1], c[2], c[3]))
                    in_channels = c[0]
                case 'M':
                    backbone.append(nn.MaxPool2D(kernel_size=2, stride=2))
                case list():
                    for i in range(c[2]):
                        backbone.append(BasicConvLayer(in_channels, c[0][0], c[0][1], c[0][2], c[0][3]))
                        backbone.append(BasicConvLayer(c[0][0], c[1][0], c[1][1], c[1][2], c[1][3]))
                        in_channels = c[1][0]
        return nn.Sequential(*backbone)

    def _build_head(self):
        return nn.Sequential(
            BasicConvLayer(1024, 1024, 3, 1, 1),
            BasicConvLayer(1024, 1024, 3, 1, 1)
        )

    def forward(self, x):
        x = self.backbone_1(x)
        y = self.backbone_passthrough(x)
        y_passthrough = self.passthrouth_conv(y)
        x = self.backbone_2(y)
        x = self.head_1(x)
        p = passthrough(y_passthrough, x)
        x = self.head_2(p)
        return x

# YOLOv1损失函数
class YoloLoss(nn.Layer):
    def __init__(self, s=MODEL_CONFIG['MODEL']['SPLIT_SIZE'], b=MODEL_CONFIG['MODEL']['NUM_BOXES'], c=MODEL_CONFIG['MODEL']['NUM_CLASSES'], lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.s = s
        self.b = b
        self.c = c
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred, target):
        ...
