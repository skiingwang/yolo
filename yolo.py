import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class YOLOv1(nn.Layer):
    def __init__(self, num_classes=20, num_bboxes=2):
        """
        Args:
            num_classes: 类别数 (论文 VOC 数据集为 20)
            num_bboxes: 每个网格预测的边界框数量 (论文中为 2)
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_bboxes = num_bboxes
        self.S = 7  # 网格大小 7x7

        # 论文中的网络结构简化实现
        # 原论文使用了 Inception 模块，这里使用标准的卷积块堆叠 (类似简化版 VGG)
        # 结构: Conv -> BN -> LeakyReLU -> MaxPool
        self.features = self._make_backbone()

        # 全连接层
        # 输入特征图大小计算: 输入 448x448
        # 经过5次下采样 -> 14x14 (如果最后步长不为2则是其他尺寸，这里用 AdaptiveAvgPool 强行适配)
        # 实际上为了简单，我们通常确保进入 FC 前的特征图尺寸固定。
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), # 假设最后特征图为 7x7x512
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.S * self.S * (self.num_bboxes * 5 + self.num_classes))
        )

    def _make_backbone(self):
        layers = []
        # 配置: (in_c, out_c, kernel, stride, padding, pool)
        # 论文 Table 1 的结构简化版
        cfg = [
            (3, 64, 7, 2, 3, True),   # 448 -> 224
            (64, 192, 3, 1, 1, True), # 224 -> 112
            (192, 128, 1, 1, 0, False),
            (128, 256, 3, 1, 1, False),
            (256, 256, 1, 1, 0, False),
            (256, 512, 3, 1, 1, True), # 112 -> 56
            # 重复 4 次 1x1, 3x3 结构
            (512, 256, 1, 1, 0, False), (256, 512, 3, 1, 1, False),
            (512, 256, 1, 1, 0, False), (256, 512, 3, 1, 1, False),
            (512, 256, 1, 1, 0, False), (256, 512, 3, 1, 1, False),
            (512, 256, 1, 1, 0, False), (256, 512, 3, 1, 1, True), # 56 -> 28
            (512, 512, 1, 1, 0, False),
            (512, 1024, 3, 1, 1, True), # 28 -> 14
            # 重复 2 次
            (1024, 512, 1, 1, 0, False), (512, 1024, 3, 1, 1, False),
            (1024, 512, 1, 1, 0, False), (512, 1024, 3, 1, 1, False),
            (1024, 1024, 3, 1, 1, False),
            (1024, 1024, 3, 2, 1, False), # 14 -> 7 (stride 2)
            (1024, 1024, 3, 1, 1, False),
            (1024, 1024, 3, 1, 1, False),
        ]

        in_channels = 3
        for layer_cfg in cfg:
            in_c, out_c, k, s, p, use_pool = layer_cfg
            if in_c != in_channels:
                continue # safety check, usually handled dynamically

            layers.append(nn.Conv2D(in_c, out_c, k, s, p))
            layers.append(nn.BatchNorm2D(out_c)) # 论文原文无BN，但现代实现通常加上以稳定训练
            layers.append(nn.LeakyReLU(0.1))

            if use_pool:
                layers.append(nn.MaxPool2D(2, 2))

            in_channels = out_c

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, 3, 448, 448]
        x = self.features(x)

        # 为了防止特征图尺寸计算误差，这里使用自适应池化确保输出为 7x7
        x = F.adaptive_avg_pool2d(x, (7, 7))

        x = self.fc(x)

        # Reshape 输出: [batch, S*S*(B*5+C)] -> [batch, S, S, B*5+C]
        x = paddle.reshape(x, [-1, self.S, self.S, self.num_bboxes * 5 + self.num_classes])

        return x
