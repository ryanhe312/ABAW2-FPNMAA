import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch

# read data
import os
import numpy as np
from PIL import Image

# utils
TYPE = ['VA_Set', 'EXPR_Set', 'AU_Set']
CLASS = [2, 1, 12]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

READERS = {
    'VA_Set': lambda path: np.genfromtxt(path, dtype=np.single, delimiter=',', skip_header=True),
    'EXPR_Set': lambda path: np.genfromtxt(path, dtype=np.int_, skip_header=True),
    'AU_Set': lambda path: np.genfromtxt(path, dtype=np.single, delimiter=',', skip_header=True)
}


# datasets
class UnifiedDataset(Dataset):
    def __init__(self,
                 idx: list,
                 image: np.ndarray,
                 label: dict,
                 img_size: int,
                 mode: str):

        # get image
        self.idx = idx
        self.image = image
        self.label = label

        # preprocess
        if mode == 'Train_Set':
            self.preprocess = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=MEAN,
                    std=STD)
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=MEAN,
                    std=STD)
            ])

    def __getitem__(self, i):
        image = Image.open(self.image[i])
        image = self.preprocess(image)

        label = [self.label['VA_Set'][i],
                 [self.label['EXPR_Set'][i]],
                 self.label['AU_Set'][i]]
        label = np.concatenate(label)

        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.idx)


class UnifiedDataModule(pl.LightningDataModule):
    def __init__(self, params: dict):
        super().__init__()

        self.batch_size = params.get('batch_size', 32)
        self.img_size = params.get('img_size', 224)
        self.num_workers = params.get('num_workers', 4)
        self.dataset_dir = params.get('dataset_dir', '../dataset/Aff-Wild/')

        with open(os.path.join(self.dataset_dir, 'file.txt')) as f:
            self.image = list(map(lambda x: os.path.join(self.dataset_dir, 'cropped_aligned', x.strip()),
                                  f.readlines()))
        self.image = np.array(self.image)

        self.label = {}
        for label_type in TYPE:
            self.label[label_type] = READERS[label_type](os.path.join(self.dataset_dir, label_type + '.txt'))

        self.index = np.arange(0, len(self.image))
        self.train_idx, self.val_idx = train_test_split(self.index, train_size=0.95, random_state=1234)

    def setup(self, stage: str = None) -> None:

        if stage == 'fit':
            self.train_dataset = UnifiedDataset(
                self.train_idx,
                self.image,
                self.label,
                self.img_size,
                'Train_Set')

            self.val_dataset = UnifiedDataset(
                self.val_idx,
                self.image,
                self.label,
                self.img_size,
                'Validation_Set')

        elif stage == 'validate':
            self.val_dataset = UnifiedDataset(
                self.val_idx,
                self.image,
                self.label,
                self.img_size,
                'Validation_Set')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

if __name__ == '__main__':
    os.chdir('..')

    dm = UnifiedDataModule({'dataset_dir':'../dataset/Aff-Wild/'})
    dm.setup('fit')
    dataloader = dm.train_dataloader()
    print(len(dataloader.dataset))
    img, label = next(iter(dataloader))
    print(img.shape, label.shape)