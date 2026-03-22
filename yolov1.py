from __future__ import annotations

from typing import TYPE_CHECKING
import paddle, paddle.nn as nn

if TYPE_CHECKING:
    from paddle import Tensor
    from typing import Tuple

__all__ = []

class BasicConvLayer(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class ConvPoolLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_basic=False, basic_incs=None, basic_outcs=None):
        super().__init__()
        self.use_basic = use_basic
        if use_basic:
            self.basic_conv = BasicConvLayer(basic_incs, basic_outcs)
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)

    def forward(self, x):
        if self.use_basic:
            x = self.basic_conv(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

"""YOLOv1模型
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

class Yolo1(nn.Layer):
    def __init__(self, num_classes=20, num_boxes=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.s = 7  # 网格大小 7x7

        self.backbone = nn.Sequential(
            # Inputs: 448x448x3
            ConvPoolLayer(3, 64, 7, 2, 3),  # 448 -> 224
            ConvPoolLayer(64, 192, 3, 1, 1),  # 224 -> 112

            BasicConvLayer(192, 128),
            nn.Conv2D(128, 256, 3, 1, 1),
            nn.LeakyReLU(),
            ConvPoolLayer(256, 512, 3, 1, 1, True, 256, 256),  # 112 -> 56

            BasicConvLayer(512, 256),
            nn.Conv2D(256, 512, 3, 1, 1),
            nn.LeakyReLU(),
            BasicConvLayer(512, 256),
            nn.Conv2D(256, 512, 3, 1, 1),
            nn.LeakyReLU(),
            BasicConvLayer(512, 256),
            nn.Conv2D(256, 512, 3, 1, 1),
            nn.LeakyReLU(),
            BasicConvLayer(512, 256),
            nn.Conv2D(256, 512, 3, 1, 1),  # 56 -> 28
            nn.LeakyReLU(),
            ConvPoolLayer(512, 1024, 3, 1, 1, True, 512, 512),  # 28 -> 14

            BasicConvLayer(1024, 512),
            nn.Conv2D(512, 1024, 3, 1, 1),
            nn.LeakyReLU(),
            BasicConvLayer(1024, 512),
            nn.Conv2D(512, 1024, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2D(1024, 1024, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2D(1024, 1024, 3, 2, 1),  # 14 -> 7
            nn.LeakyReLU(),

            nn.Conv2D(1024, 1024, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2D(1024, 1024, 3, 1, 1),  # 7 -> 7
            nn.LeakyReLU()
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(4096, self.s * self.s * (self.num_boxes * 5 + self.num_classes))  # 1470
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x.reshape([-1, self.s, self.s, self.num_boxes * 5 + self.num_classes])  # [batch, 7, 7, 30]

# YOLOv1损失函数
class YoloLoss(nn.Layer):
    def __init__(self, s=7, b=2, c=20, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.s = s
        self.b = b
        self.c = c
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred, target):
        # pred: [batch, 7, 7, b*5+c]
        pred_box = pred[:, :, :, :self.b*5].reshape([-1, self.s, self.s, self.b, 5])  # [batch, 7, 7, 2, 5]
        pred_cls = pred[:, :, :, self.b*5:].reshape([-1, self.s, self.s, self.c])  # [batch, 7, 7, 20]

        # target: [batch, 7, 7, 5]
        target_box = target[:, :, :, :self.b*5].reshape([-1, self.s, self.s, self.b, 5])  # [batch, 7, 7, 2, 5]
        target_cls = target[:, :, :, self.b*5:].reshape([-1, self.s, self.s, self.c])  # [batch, 7, 7, 20]

        pred_x = pred_box[:, :, :, :, 0]
        pred_y = pred_box[:, :, :, :, 1]
        pred_w = pred_box[:, :, :, :, 2]
        pred_h = pred_box[:, :, :, :, 3]

        target_x = target_box[:, :, :, :, 0]
        target_y = target_box[:, :, :, :, 1]
        target_w = target_box[:, :, :, :, 2]
        target_h = target_box[:, :, :, :, 3]

        center_loss = paddle.mean((pred_x - target_x) ** 2) + paddle.mean((pred_y - target_y) ** 2)
        wh_loss = paddle.mean((paddle.sqrt(paddle.abs(pred_w)) - paddle.sqrt(paddle.abs(target_w))) ** 2) + paddle.mean((paddle.sqrt(paddle.abs(pred_h)) - paddle.sqrt(paddle.abs(target_h))) ** 2)
        coord_loss = self.lambda_coord * (center_loss + wh_loss)

        pred_conf = pred_box[:, :, :, :, 4]
        target_conf = target_box[:, :, :, :, 4]
        conf_mask_obj = (target_conf == 1).astype('float32')
        conf_mask_noobj = (target_conf == 0).astype('float32')
        conf_loss_obj = paddle.mean(((pred_conf - target_conf) ** 2) * conf_mask_obj)
        conf_loss_noobj = self.lambda_noobj * paddle.mean(((pred_conf - target_conf) ** 2) * conf_mask_noobj)
        conf_loss = conf_loss_obj + conf_loss_noobj

        cls_mask_obj = (paddle.max(target_conf, axis=-1) == 1).astype('float32').unsqueeze(-1)  # [batch, 7, 7, 1]
        cls_loss = paddle.mean(((pred_cls - target_cls) ** 2) * cls_mask_obj)

        total_loss = coord_loss + conf_loss + cls_loss
        return total_loss

