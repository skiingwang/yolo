from __future__ import annotations

from typing import TYPE_CHECKING
from utils import read_yaml, passthrough, calc_iou
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


# YOLOv2损失函数
class YoloV2Loss(nn.Layer):
    def __init__(self, anchors, num_classes=10, lambda_coord=MODEL_CONFIG['LOSS']['LAMBDA_COORD'],
                 lambda_noobj=MODEL_CONFIG['LOSS']['LAMBDA_NOOBJ'], lambda_obj=MODEL_CONFIG['LOSS']['LAMBDA_OBJ'],
                 lambda_class=MODEL_CONFIG['LOSS']['LAMBDA_CLASS'], lambda_prior=MODEL_CONFIG['LOSS']['LAMBDA_PRIOR']):
        super().__init__()
        self.anchors = paddle.to_tensor(anchors, dtype='float32')  # 锚框 [w, h]
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class
        self.lambda_prior = lambda_prior

    def forward(self, preds, labels):
        """
        preds: [N, 125, 13, 13] (假设输入是 5 个框，10 类)，125 = 5 * (4 coords + 1 conf + 10 classes)
        labels: [N, 13, 13, 5 + C]，前 4 个是 box (x,y,w,h), 第5个是 conf (0或1), 后面是 classes
        """
        batch_size, grid_size = preds.shape[0], preds.shape[2]

        # 1. 预测值 (Reshape Predictions)
        # 从 [N, 125, 13, 13] -> [N, 5, 25, 13, 13]
        preds = preds.transpose([0, 2, 3, 1]).reshape([batch_size, grid_size, grid_size, self.num_anchors, 4 + 1 + self.num_classes])

        pred_boxes, pred_confs, pred_cls = preds[..., 0:4], preds[..., 4:5], preds[..., 5:]

        # 2. 标签 (Reshape Labels)
        # labels: [N, 13, 13, 5 + C]
        label_boxes, label_confs, label_cls = labels[..., 0:4], labels[..., 4:5], labels[..., 5:]

        # 3. 计算 IoU 并分配责任 (Assign Responsibility)
        # pred_boxes: [N, 13, 13, 5, 4]
        # label_boxes: [N, 13, 13, 4] -> [N, 13, 13, 1, 1, 4]
        label_boxes_exp = label_boxes.unsqueeze(3).unsqueeze(3)

        # 计算所有锚框与所有真实框的 IoU
        # iou: [N, 13, 13, 5, 13, 13]
        iou = calc_iou(pred_boxes, label_boxes_exp)

        # 对于每个真实框 (GT)，找到 IoU 最大的那个锚框
        # max_iou_per_gt: [N, 13, 13] (该网格内最高的 IoU)
        # best_anchor_idx: [N, 13, 13] (负责该 GT 的锚框索引 0-4)
        max_iou, best_anchor_idx = paddle.max(iou, axis=3)  # 在锚框维度取最大
        max_iou = max_iou.squeeze(3)  # [N, 13, 13]
        best_anchor_idx = best_anchor_idx.squeeze(3)  # [N, 13, 13]

        # responsible_mask: [N, 13, 13, 5]，如果某个锚框是负责该网格内物体的，则为 1
        responsible_mask = paddle.zeros_like(pred_confs)
        # 使用scatter或者循环填充，这里用简单的索引赋值，best_anchor_idx是标量索引，需要扩展维度
        idx = paddle.arange(grid_size).unsqueeze(0).expand([batch_size, grid_size])  # [N, 13]
        # 构造索引 [N, 13, 13, 1]
        row_idx = idx.unsqueeze(2).expand([batch_size, grid_size, grid_size, 1])
        col_idx = idx.unsqueeze(1).expand([batch_size, grid_size, grid_size, 1])

        # indices: [N, 13, 13, 1]
        indices = best_anchor_idx.unsqueeze(3)
        # values: [N, 13, 13, 1] 全 1
        values = paddle.ones_like(indices, dtype='float32')
        # 使用 scatter_ 填充
        # 注意：scatter_ 需要 indices 是 int64
        responsible_mask = paddle.zeros([batch_size, grid_size, grid_size, self.num_anchors], dtype='float32')
        responsible_mask = paddle.scatter_(responsible_mask, 3, indices.astype('int64'), values)

        # 4. 计算损失
        # 坐标损失 (Coordinate Loss)
        # 仅对负责的框计算 (responsible_mask == 1)，提取负责框的预测坐标和真实坐标
        # pred_boxes_resp: [N, 13, 13, 4]
        pred_boxes_resp = paddle.sum(pred_boxes * responsible_mask, axis=3)

        # 真实坐标 (只取负责的那个 GT 的坐标)
        pred_wh = paddle.sqrt(paddle.abs(pred_boxes_resp[..., 2:4]) + 1e-6)
        label_wh = paddle.sqrt(paddle.abs(label_boxes[..., 2:4]) + 1e-6)

        # 坐标损失 (x, y, w, h)
        coord_loss = self.mse_loss(pred_boxes_resp[..., 0:2], label_boxes[..., 0:2]) + self.mse_loss(pred_wh, label_wh)
        coord_loss *= self.lambda_coord

        # 1. 有物体的置信度损失 (Object Confidence Loss)
        # 公式：lambda_obj * (IOU_truth - b_o)^2
        iou_target = max_iou.unsqueeze(3)  # [N, 13, 13, 1]
        obj_conf_loss = self.mse_loss(pred_confs * responsible_mask, iou_target * responsible_mask)
        obj_conf_loss *= self.lambda_obj

        # 2. 无物体的置信度损失 (No Object Confidence Loss) + 先验损失 (Prior Loss)
        # 条件：Max IoU < Thresh (通常设为 0.6) 或者 simply not responsible
        # 定义负样本掩码：不是负责框，且 IoU 较低 (或者简单地，不是负责框)
        noobj_mask = 1 - responsible_mask

        # 置信度部分：目标是 0
        conf_noobj_loss = self.mse_loss(pred_confs * noobj_mask, paddle.zeros_like(pred_confs))
        conf_noobj_loss *= self.lambda_noobj

        # 扩展 anchors 到 [N, 13, 13, 5, 4]
        prior_boxes = self.anchors.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
            [batch_size, grid_size, grid_size, self.num_anchors, 4])

        # 仅计算负样本的 prior loss
        prior_loss = self.mse_loss(pred_boxes * noobj_mask, prior_boxes * noobj_mask)
        prior_loss *= self.lambda_prior

        total_conf_loss = conf_noobj_loss + prior_loss

        # 类别损失 (Class Loss)
        # 仅对负责的框计算
        cls_loss = self.mse_loss(pred_cls * responsible_mask, label_cls.unsqueeze(3) * responsible_mask)
        cls_loss *= self.lambda_class

        # 总损失
        total_loss = coord_loss + obj_conf_loss + total_conf_loss + cls_loss

        return total_loss
