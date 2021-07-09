import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms

# read data
import os
import pandas
import pickle
import numpy as np
from PIL import Image

# utils
TYPE = ['VA_Set', 'EXPR_Set', 'AU_Set']
CLASS = [2, 7, 12]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ANNOTATION_PATH = {
    'VA_Set': 'mixedAnnotation/mixed_VA_annotations.pkl',
    'EXPR_Set': 'mixedAnnotation/mixed_EXPR_annotations.pkl',
    'AU_Set': 'mixedAnnotation/mixed_AU_annotations.pkl'
}


# datasets
class BalancedDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 img_size: int,
                 type: str,
                 mode: str):

        annotation_path = os.path.join(dataset_dir, ANNOTATION_PATH[type])
        annotation = pickle.load(open(annotation_path, 'rb'))
        self.image = annotation[mode]['path']
        self.label = annotation[mode]['label']
        if type == 'EXPR_Set':
            self.label = self.label.astype(np.int_)
        else:
            self.label = self.label.astype(np.single)

        # preprocess
        if mode == 'Train_Set':
            self.preprocess = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=MEAN,
                    std=STD)
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=MEAN,
                    std=STD)
            ])

    def __getitem__(self, i):
        image = Image.open(self.image[i])
        image = self.preprocess(image)
        label = self.label[i]

        return image, label

    def __len__(self):
        return len(self.image)


# datamodules
class BalancedDataModule(pl.LightningDataModule):

    def __init__(self, params: dict):
        super().__init__()

        self.batch_size = params.get('batch_size', 32)
        self.img_size = params.get('img_size', 224)
        self.data_type = params.get('data_type', 'VA_Set')
        self.num_workers = params.get('num_workers', 4)
        self.dataset_dir = params.get('dataset_dir', '../dataset/Aff-Wild/')

    def setup(self, stage:str = None) -> None:

        if stage == 'fit':
            self.train_dataset = BalancedDataset(
                self.dataset_dir,
                self.img_size,
                self.data_type,
                'Train_Set')

            self.val_dataset = BalancedDataset(
                self.dataset_dir,
                self.img_size,
                self.data_type,
                'Validation_Set')

        elif stage == 'validate':
            self.val_dataset = BalancedDataset(
                self.dataset_dir,
                self.img_size,
                self.data_type,
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

    dm = BalancedDataModule({'dataset_dir':'../dataset/Aff-Wild/','num_workers':128 , 'data_type':'AU_Set'})
    dm.setup('fit')
    dataloader = dm.train_dataloader()
    print(len(dataloader.dataset))
    img, label = next(iter(dataloader))
    print(img.shape, label.shape)
