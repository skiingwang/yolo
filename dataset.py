import os, numpy as np, paddle, pathlib
from PIL import Image
from paddle.io import Dataset
from utils import read_yaml

DATASET_CONFIG = read_yaml('./config.yaml')

class YoloDataset(Dataset):
    def __init__(self, data_path=DATASET_CONFIG['DATASET']['PATH'], fmt='txt', split='train', transform=None):
        super().__init__()
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
        # img = np.transpose(img, (2, 1, 0))
        labels, boxes = self._label_processor(label_file)
        if self.transform:
            return self.transform(img, labels, boxes)
        labels = np.column_stack((labels, boxes))
        return img, labels

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = [paddle.to_tensor(img) if not isinstance(img, paddle.Tensor) else img for img in imgs]
    imgs = paddle.stack(imgs, axis=0)
    return imgs, paddle.to_tensor(labels)


def preprocessor(labels, boxes):
    s, b, c = DATASET_CONFIG['MODEL']['SPLIT_SIZE'], DATASET_CONFIG['MODEL']['NUM_BOXES'], DATASET_CONFIG['MODEL']['NUM_CLASSES']

    if not isinstance(labels, paddle.Tensor):
        labels = paddle.to_tensor(labels)
    if not isinstance(boxes, paddle.Tensor):
        boxes = paddle.to_tensor(boxes)

    targets = paddle.zeros([s, s, b * 5 + c])

    for obj in range(labels.shape[0]):  # 遍历每个样本的所有物体
        x, y, w, h = boxes[obj]
        cls = labels[obj]

        grid_i, grid_j = int(y * s), int(x * s)  # 物体所属网格索引
        x_grid, y_grid = x * s - grid_j, y * s - grid_i  # 计算锚框的中心坐标相对于网格的坐标（局部归一化坐标）
        w_grid, h_grid = w * s, h * s  # w * s 代表目标物体的宽度以网格尺寸为单位的相对坐标（局部归一化尺寸）

        if targets[grid_i, grid_j, 0] == 0:
            targets[grid_i, grid_j, 0] = 1  # conf=1（表示有物体）
            targets[grid_i, grid_j, 1:5] = paddle.tensor([x_grid, y_grid, w_grid, h_grid])  # 填充锚框的中心坐标和尺寸（局部归一化边界框）
            targets[grid_i, grid_j, b*5+cls] = 1  # 对应类别的位置设为1

    return targets  # [s, s, [conf, x, y, w, h, cls1, cls2, ...]]

def img_scale(w, h):
    return paddle.vision.transforms.Compose([
        paddle.vision.transforms.Resize((w, h)),
        paddle.vision.transforms.ToTensor(),
    ])

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels, boxes):
            image = self.transforms[0](img)  # [3, height, width]
            image = paddle.transpose(image, [0, 2, 1])
            label = self.transforms[1](labels, boxes)
            return image, label