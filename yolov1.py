from __future__ import annotations

from typing import TYPE_CHECKING
from utils import read_yaml, calc_iou
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
    def __init__(self, in_channels=3, split_size=MODEL_CONFIG['MODEL']['SPLIT_SIZE'], num_classes=MODEL_CONFIG['MODEL']['NUM_CLASSES'], num_boxes=MODEL_CONFIG['MODEL']['NUM_BOXES']):
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
        return x.reshape([-1, self.s, self.s, self.num_boxes * 5 + self.num_classes])  # [batch, 7, 7, SBC]

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
    def __init__(self, s=MODEL_CONFIG['MODEL']['SPLIT_SIZE'], b=MODEL_CONFIG['MODEL']['NUM_BOXES'], c=MODEL_CONFIG['MODEL']['NUM_CLASSES'], lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.s = s
        self.b = b
        self.c = c
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.sse = nn.MSELoss(reduction='sum')

    def forward(self, preds, labels):

        # [s, s, SBC]
        exist_obj, labels_box, labels_cls = labels[..., 0:1], labels[..., 1:5], labels[..., 10:]
        pred_box1, pred_box2, pred_cls = preds[..., 1:5], preds[..., 5:10], preds[..., 10:]

        # 筛选IOU更大的预测框（x,y,w,h）值：[s, s, 4]
        iou1, iou2 = calc_iou(pred_box1, labels_box), calc_iou(pred_box2, labels_box)
        max_iou = paddle.maximum(iou1, iou2).unsqueeze(2)  # [s, s, 1]
        iou_compare = (iou1 > iou2).astype('float32').unsqueeze(2)  # [s, s, 1(bool)]，转换后，1为pred_box1，0为pred_box2
        pred_box = pred_box1 * iou_compare + pred_box2 * (1 - iou_compare)

        # 筛选有物体的Grid：[s, s, 4]
        grid_obj_box = pred_box * exist_obj

        # 计算中心坐标损失：[s, s, 1]
        center_loss = (self.lambda_coord * self.sse(grid_obj_box[..., 1], labels_box[..., 1]) +
                       self.sse(grid_obj_box[..., 2], labels_box[..., 2]))

        # 计算宽度和高度损失：[s, s, 1]
        wh_loss = (self.lambda_coord * self.sse(paddle.sign(labels_box[..., 3]) * paddle.sqrt(paddle.abs(grid_obj_box[..., 3]) + 1e-6),  labels_box[..., 3]) +
                   self.sse(paddle.sign(labels_box[..., 4]) * paddle.sqrt(paddle.abs(grid_obj_box[..., 4]) + 1e-6), labels_box[..., 4]))

        # 计算坐标损失：[s, s, 1]
        coord_loss = center_loss + wh_loss

        # 计算有物体的Grid置信度损失：[s, s, 1]
        pred_conf1, pred_conf2 = preds[..., 4:5], preds[..., 9:10]
        pred_conf = pred_conf1 * iou_compare + pred_conf2 * (1 - iou_compare)

        pred_grid_obj_conf = pred_conf * exist_obj
        gt_grid_obj_conf = max_iou * exist_obj

        obj_conf_loss = self.sse(pred_grid_obj_conf, gt_grid_obj_conf)

        # 计算无物体的Grid置信度损失：[s, s, 1]
        exist_no_obj = 1 - exist_obj
        pred_grid_no_obj_conf1, pred_grid_no_obj_conf2 = pred_conf1 * exist_no_obj, pred_conf2 * exist_no_obj
        label_grid_no_obj_conf = paddle.zeros_like(pred_grid_no_obj_conf1)
        no_obj_conf_loss = self.lambda_noobj * (self.sse(pred_grid_no_obj_conf1, label_grid_no_obj_conf) + self.sse(pred_grid_no_obj_conf2, label_grid_no_obj_conf))

        # 计算置信度损失：[s, s, 1]
        conf_loss = obj_conf_loss + no_obj_conf_loss

        # 计算类别损失
        pred_obj_cls = exist_obj * pred_cls
        cls_loss = self.sse(pred_obj_cls, labels_cls)

        return coord_loss + conf_loss + cls_loss
