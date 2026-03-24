import os, numpy as np, paddle, pathlib
from PIL import Image
from paddle.io import Dataset


class YoloDataset(Dataset):
    def __init__(self, data_path, fmt='txt', split='train', transform=None):
        self.data_path = data_path
        self.fmt = fmt
        self.transform = transform
        match split:
            case 'train':
                self.imgs_dir = pathlib.Path(os.path.join(data_path, 'images', 'train'))
                self.labels_dir = pathlib.Path(os.path.join(data_path, 'labels', 'train'))
            case 'test':
                self.imgs_dir = pathlib.Path(os.path.join(data_path, 'images', 'test'))
                self.labels_dir = pathlib.Path(os.path.join(data_path, 'labels', 'test'))
            case _:
                raise ValueError(f'Unknown split {split}')
        self.imgs = sorted(list(self.imgs_dir.glob('*.*')))
        self.labels = sorted(list(self.labels_dir.glob('*.*')))

        try:
            if not len(self.imgs) == len(self.labels):
                raise ValueError(f'Number of images ({len(self.imgs)}) does not match number of labels ({len(self.labels)})')
        except ValueError as e:
            print(e)

    def _label_processor(self, label_file):
        labels, boxes = [], []
        if not label_file.exists():
            raise FileNotFoundError(f'Label file {label_file} does not exist')

        match self.fmt:
            case 'txt':
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(' ')
                        if len(parts) == 5:
                            # [类别，全局归一化x坐标，全局归一化y坐标，全局归一化宽度，全局归一化高度]
                            cls, x, y, w, h = map(float, parts)
                            labels.append(int(cls))
                            boxes.append([x, y, w, h])
                return np.int64(labels), np.float32(boxes)

    def __getitem__(self, idx):
        img_file, label_file = self.imgs[idx], self.labels[idx]
        img = np.array(Image.open(img_file).convert('RGB'))
        if self.transform:
            img = self.transform(img)
        labels, boxes = self._label_processor(label_file)
        return img, labels, boxes

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    imgs, labels, boxes = zip(*batch)
    imgs = paddle.stack(imgs, axis=0)
    return imgs, labels, boxes


def _iou_calc(pred_box, gt_box):
    pred_x, pred_y = pred_box[0], pred_box[1]
    pred_w, pred_h = pred_box[2], pred_box[3]
    gt_x, gt_y = gt_box[0], gt_box[1]
    gt_w, gt_h = gt_box[2], gt_box[3]

    # 计算交集的左上角和右下角的x,y坐标
    inter_x1 = paddle.max(paddle.to_tensor([pred_x - pred_w / 2, gt_x - gt_w / 2]))
    inter_y1 = paddle.max(paddle.to_tensor([pred_y - pred_h / 2, gt_y - gt_h / 2]))
    inter_x2 = paddle.min(paddle.to_tensor([pred_x + pred_w / 2, gt_x + gt_w / 2]))
    inter_y2 = paddle.min(paddle.to_tensor([pred_y + pred_h / 2, gt_y + gt_h / 2]))

    # 计算交集的面积
    inter_w = paddle.max(paddle.to_tensor([0, inter_x2 - inter_x1]))
    inter_h = paddle.max(paddle.to_tensor([0, inter_y2 - inter_y1]))
    inter_area = inter_w * inter_h

    # 计算并集的面积
    pred_area = pred_w * pred_h
    gt_area = gt_w * gt_h
    union_area = pred_area + gt_area - inter_area

    return inter_area / (union_area + 1e-10)

def preprocessor(labels, boxes, s, b, c, pred_boxes):
    # pred_boxes: [batch, s, s, b * 5 + c]
    device = 'cuda' if paddle.cuda.is_available() else 'cpu'
    batch_size = len(labels)
    targets = paddle.zeros([batch_size, s, s, b * 5 + c]).to(device)

    # 遍历每个样本， 获取类别和锚框坐标
    for i in range(batch_size):
        if len(labels[i]) == 0:  # 当前样本无物体则跳过
            continue
        sample_labels, sample_boxes = labels[i], boxes[i]
        if not isinstance(sample_labels, paddle.Tensor):
            sample_labels = paddle.to_tensor(sample_labels)
        if not isinstance(sample_boxes, paddle.Tensor):
            sample_boxes = paddle.to_tensor(sample_boxes)

        sample_labels, sample_boxes = sample_labels.to(device), sample_boxes.to(device)

        for j in range(sample_labels.shape[0]):  # 遍历每个样本的所有物体
            cls = sample_labels[j].item()
            x, y, w, h = sample_boxes[j]

            """物体所属网格索引
            绝对坐标列索引：grid_x = int(center_x / grid_w), grid_w = w / s
            相对坐标列索引：grid_x = int(x / (grid_w / w)) = int(x / (1 / s)), x = center_x / w
            """
            grid_x, grid_y = int(x * s), int(y * s)

            if grid_x >= s: grid_x = s-1
            if grid_y >= s: grid_y = s-1

            # 计算锚框的中心坐标相对于网格的坐标（局部归一化坐标）
            x_grid, y_grid = x * s - grid_x, y * s - grid_y

            # 根据IOU选择边界框预测器
            iou_1 = _iou_calc(pred_boxes[i, grid_y, grid_x, :4], paddle.tensor([x, y, w, h]))
            iou_2 = _iou_calc(pred_boxes[i, grid_y, grid_x, 5:9], paddle.tensor([x, y, w, h]))
            best_pred_idx = int(paddle.argmax(paddle.to_tensor([iou_1, iou_2])).item())  # 选择IOU最大的锚框的索引

            # 填充边界框部分（x, y, w, h, conf）
            targets[i, grid_y, grid_x, best_pred_idx * 5 : best_pred_idx * 5 + 4] = paddle.tensor([x_grid, y_grid, w, h])
            targets[i, grid_y, grid_x, best_pred_idx * 5 + 4] = 1.0  # conf=1（表示有物体）

            # 填充类别部分（one-hot编码）
            targets[i, grid_y, grid_x, b * 5 + cls] = 1.0  # 对应类别的位置设为1

    return targets