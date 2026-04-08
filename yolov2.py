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
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2D(out_channels)
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
        self.s = split_size  # 网格大小 7x7
        self.network_cfg = [
            (32, 3, 1, 1),
            'M',
            (64, 3, 1, 1),
            'M',
            (128, 3, 1, 1),
            (64, 1, 1, 0),
            (128, 3, 1, 1),
            'M',
            (256, 3, 1, 1),
            (128, 1, 1, 0),
            (256, 3, 1, 1),
            'M',
            [
                (512, 3, 1, 1),
                (256, 1, 1, 0),
                2
            ],
            (512, 3, 1, 1),
            'M',
            [
                (1024, 3, 1, 1),
                (512, 1, 1, 0),
                2
            ],
            (1024, 3, 1, 1),
        ]
        self.backbone = self._build_backbone(self.network_cfg, in_channels)
        self.head = self._build_head()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def _build_backbone(self, cfg, in_channels):
        backbone = []
        for c in cfg:
            match c:
                case tuple():
                    backbone.append(BasicConvLayer(in_channels, c[0], c[1], c[2], c[3]))
                    in_channels = c[0]
                case str():
                    backbone.append(nn.MaxPool2D(kernel_size=2, stride=2))
                case list():
                    for _ in range(c[2]):
                        backbone.append(BasicConvLayer(in_channels, c[0][0], c[0][1], c[0][2], c[0][3]))
                        backbone.append(BasicConvLayer(c[0][0], c[1][0], c[1][1], c[1][2], c[1][3]))
                        in_channels = c[1][0]
        return nn.Sequential(*backbone)

    def _build_head(self):
        return nn.Sequential(
            nn.Conv2D(1024, self.num_classes, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

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

