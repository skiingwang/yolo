from __future__ import annotations

from typing import TYPE_CHECKING
import paddle, paddle.nn as nn

if TYPE_CHECKING:
    from paddle import Tensor
    from typing import Tuple

__all__ = []

class BasicConvLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
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
    def __init__(self, in_channels=3, split_size=7, num_classes=20, num_boxes=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.s = split_size  # 网格大小 7x7
        self.network_cfg = [
            (64, 7, 2, 3),
            'M',
            (192, 3, 1, 1),
            'M',
            (128, 1, 1, 0),
            (256, 3, 1, 1),
            (256, 1, 1, 0),
            (512, 3, 1, 1),
            'M',
            [
                (256, 1, 1, 0),
                (512, 3, 1, 1),
                4
            ],
            (512, 1, 1, 0),
            (1024, 3, 1, 1),
            'M',
            [
                (512, 1, 1, 0),
                (1024, 3, 1, 1),
                2
            ],
            (1024, 3, 1, 1),
            (1024, 3, 2, 1),
            (1024, 3, 1, 1),
            (1024, 3, 1, 1),
        ]
        self.backbone = self._build_backbone(self.network_cfg, in_channels)
        self.head = self._build_head()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x.reshape([-1, self.s, self.s, self.num_boxes * 5 + self.num_classes])  # [batch, 7, 7, 30]

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
            nn.Flatten(),
            nn.Linear(1024 * self.s * self.s, 4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.s * self.s * (self.num_boxes * 5 + self.num_classes))  # 1470
        )

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
        pred_box = pred[..., :self.b*5].reshape([-1, self.s, self.s, self.b, 5])  # [batch, 7, 7, 2, 5]
        pred_cls = pred[..., self.b*5:]  # [batch, 7, 7, 20]
        # target: [batch, 7, 7, b*5+c]
        target_box = target[..., :self.b*5].reshape([-1, self.s, self.s, self.b, 5])  # [batch, 7, 7, 2, 5]
        target_cls = target[..., self.b*5:]  # [batch, 7, 7, 20]

        # [batch, 7, 7, 2]
        pred_x, pred_y = pred_box[..., 0], pred_box[..., 1]
        pred_w, pred_h = pred_box[..., 2], pred_box[..., 3]
        # [batch, 7, 7, 2]
        target_x, target_y = target_box[..., 0], target_box[..., 1]
        target_w, target_h = target_box[..., 2], target_box[..., 3]
        # [batch, 7, 7, 2]
        pred_conf, target_conf = pred_box[..., 4], target_box[..., 4]

        # 计算中心坐标损失：[batch, 7, 7, 2]
        center_loss = self.lambda_coord * (((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2)).sum()
        # 计算宽度和高度损失：[batch, 7, 7, 2]
        wh_loss = self.lambda_coord * (((paddle.sqrt(paddle.abs(pred_w)) - paddle.sqrt(target_w)) ** 2 + (paddle.sqrt(paddle.abs(pred_h)) - paddle.sqrt(target_h)) ** 2)).sum()
        # 计算坐标损失：
        coord_loss = center_loss + wh_loss
        # 计算置信度损失：mask_obj.shape=[batch, 7, 7, 2], mask_obj:paddle.Tensor[bool]
        mask_obj, mask_noobj = target_conf > 0, target_conf == 0
        # target_conf[mask_obj]：过滤出有物体的Grid，维度=标注框数量
        conf_obj_loss = ((pred_conf[mask_obj] - target_conf[mask_obj]) ** 2).sum()
        # target_conf[mask_noobj]：过滤出无物体的Grid，维度=7*7-标注框数量
        conf_noobj_loss = self.lambda_noobj * ((pred_conf[mask_noobj] - target_conf[mask_noobj]) ** 2).sum()
        conf_loss = conf_obj_loss + conf_noobj_loss

        # 计算类别损失
        cls_mask_obj = mask_obj.any(axis=-1)  #每个Grid中，B个预测框只要有一个有物体，则计算CLS损失，cls_mask_obj为True，[batch, 7, 7]
        cls_loss = ((pred_cls[cls_mask_obj] - target_cls[cls_mask_obj])**2).sum()

        total_loss = coord_loss + conf_loss + cls_loss
        print(f'total_loss={total_loss}')
        return total_loss

