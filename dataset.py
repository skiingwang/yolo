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