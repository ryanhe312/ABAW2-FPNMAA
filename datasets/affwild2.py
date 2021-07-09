import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms
import torch

# read data
import os
import numpy as np
from PIL import Image

# utils
TYPE = ['VA_Set', 'EXPR_Set', 'AU_Set']
CLASS = [2, 7, 12]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
READERS = {
    'VA_Set': lambda path: np.genfromtxt(path, dtype=np.single, delimiter=',', skip_header=True),
    'EXPR_Set': lambda path: np.genfromtxt(path, dtype=np.int_, skip_header=True),
    'AU_Set': lambda path: np.genfromtxt(path, dtype=np.single, delimiter=',', skip_header=True)
}

TEST_SET = {
    'VA_Set': 'test_set_VA_Challenge_frame.txt',
    'EXPR_Set': 'test_set_Expr_Challenge_frame.txt',
    'AU_Set': 'test_set_AU_Challenge_frame.txt'
}

class Cropped_Aligned_Predict(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 img_size: int,
                 type: str,
                 test_set: bool):
        data_dir = os.path.join(dataset_dir, 'cropped_aligned')

        self.test_set = {}
        if test_set:
            with open(os.path.join(dataset_dir, TEST_SET[type])) as f:
                lines = [line.strip().split() for line in f.readlines()]
            self.test_set = {line[0]:int(line[1]) for line in lines}

        self.image = []
        if self.test_set:
            for video in self.test_set.keys():
                for root, dirs, files in os.walk(os.path.join(data_dir,video)):
                    for name in files:
                        if 'jpg' in name and '3d' not in name:
                        # if 'jpg' in name and '3d' in name:
                            self.image.append(os.path.join(root, name))
        else:
            for root, dirs, files in os.walk(data_dir):
                for name in files:
                    if 'jpg' in name and '3d' not in name:
                    # if 'jpg' in name and '3d' in name:
                        self.image.append(os.path.join(root, name))

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

        return image

    def __len__(self):
        return len(self.image)

# datasets
class Cropped_Aligned_Fit(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 img_size: int,
                 type: str,
                 mode: str):

        # max label length
        max_length = 5000

        # get data_dir
        data_dir = os.path.join(dataset_dir, 'cropped_aligned')
        annotation_dir = os.path.join(dataset_dir, 'annotations')

        # get list
        label_dir = os.path.join(annotation_dir, type, mode)
        video_list = filter(lambda x: 'txt' in x, os.listdir(label_dir))

        # get image
        self.image = []
        self.label = []
        for video in video_list:
            label = READERS[type](os.path.join(label_dir, video)).squeeze()
            frames = list(filter(lambda x: 'jpg' in x and '3d' not in x, os.listdir(os.path.join(data_dir, video[:-4]))))
            frame_numbers = {int(frame[:-4])-1: frame for frame in frames}

            # frames = list(filter(lambda x: 'jpg' in x and '3d' in x, os.listdir(os.path.join(data_dir, video[:-4]))))
            # frame_numbers = {int(frame[:-7])-1: frame for frame in frames}

            if type == 'EXPR_Set':
                label_mask = np.intersect1d(np.argwhere(label >= 0), list(frame_numbers.keys()))
            elif type == 'AU_Set':
                label_mask = np.intersect1d(np.argwhere((label >= 0).all(axis=-1)), list(frame_numbers.keys()))
            else:
                label_mask = list(frame_numbers.keys())

            # label_mask = np.random.choice(label_mask, min(max_length, len(label_mask)), replace=False)

            self.label.append(label[label_mask])
            self.image += [os.path.join(data_dir, video[:-4], frame_numbers[frame]) for frame in label_mask]
        self.label = np.concatenate(self.label)

        # sanity check
        assert len(self.image) == len(self.label)

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
        # image_3d = Image.open(self.image[i][:-4]+'_3d.jpg')
        image = self.preprocess(image)
        # image_3d = self.preprocess(image_3d)
        label = self.label[i]

        return image, label
        # return torch.stack([image,image_3d]), label

    def __len__(self):
        return len(self.image)

# datamodules
class AffWild2DataModule(pl.LightningDataModule):

    def __init__(self, params: dict):
        super().__init__()

        self.batch_size = params.get('batch_size', 32)
        self.img_size = params.get('img_size', 224)
        self.data_type = params.get('data_type', 'VA_Set')
        self.num_workers = params.get('num_workers', 4)
        self.dataset_dir = params.get('dataset_dir', '../dataset/Aff-Wild/')
        self.test_set = params.get('test_set', False)

    def setup(self, stage:str = None) -> None:

        if stage == 'predict':
            self.predict_dataset = Cropped_Aligned_Predict(
                self.dataset_dir,
                self.img_size,
                self.data_type,
                self.test_set)

            self.train_dataset = Cropped_Aligned_Fit(
                self.dataset_dir,
                self.img_size,
                self.data_type,
                'Train_Set')

        elif stage == 'fit':
            self.train_dataset = Cropped_Aligned_Fit(
                self.dataset_dir,
                self.img_size,
                self.data_type,
                'Train_Set')

            self.val_dataset = Cropped_Aligned_Fit(
                self.dataset_dir,
                self.img_size,
                self.data_type,
                'Validation_Set')

        elif stage == 'validate':
            self.val_dataset = Cropped_Aligned_Fit(
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

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

if __name__ == '__main__':
    os.chdir('..')

    dm = AffWild2DataModule({'dataset_dir':'../dataset/Aff-Wild/', 'data_type': 'EXPR_Set', 'test_set':True})
    dm.setup('predict')
    dataloader = dm.predict_dataloader()
    print(len(dataloader.dataset))
    img = next(iter(dataloader))
    print(img.shape)
