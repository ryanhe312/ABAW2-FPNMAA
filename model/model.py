import torch
import numpy as np
import torchvision
import torch.nn as nn
import torchmetrics as tm
import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.nn import functional as F

from .network import *
from .metrics import *

class MonoClassifier(pl.LightningModule):
    def __init__(self, params: dict):
        super().__init__()
        self.save_hyperparameters(params)

        self.backbone = torchvision.models.mobilenet_v2(pretrained=True)

        if self.hparams.data_type == 'VA_Set':
            self.head = ClassificationHead(input_dim=1000, target_dim=2)
            self.loss = nn.MSELoss()
        elif self.hparams.data_type == 'EXPR_Set':
            self.head = ClassificationHead(input_dim=1000, target_dim=7)
            self.loss = nn.CrossEntropyLoss()
            # self.loss = lambda x, y: F.cross_entropy(x, y) + F.multi_margin_loss(x, y)
        else:
            self.head = ClassificationHead(input_dim=1000, target_dim=12)
            self.loss = nn.BCEWithLogitsLoss()
            # self.loss = lambda x, y: F.binary_cross_entropy_with_logits(x, y) + F.l1_loss(x, y)

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        y_hat = self.head(embedding)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        y_hat = y_hat.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        self.log_dict({'val/loss': loss})

        return y_hat, y

    def validation_epoch_end(self, outputs) -> None:
        y_hat = []
        y = []

        for step in outputs:
            y_hat.append(step[0])
            y.append(step[1])

        y_hat = np.concatenate(y_hat, axis=0)
        y = np.concatenate(y, axis=0)

        if self.hparams.data_type == 'VA_Set':
            item, sum = VA_metric(y_hat, y)
            self.log_dict({'val/CCC-V': item[0], 'val/CCC-A': item[1], 'val/score': sum})

        elif self.hparams.data_type == 'EXPR_Set':
            f1_acc, score, matrix = EXPR_metric(y_hat, y)
            self.log_dict({'val/f1': f1_acc[0], 'val/acc': f1_acc[1], 'val/score': score})

            fig, ax = plt.subplots()
            ax = sns.heatmap(matrix, cmap='Blues')
            self.logger.experiment.add_figure('val/conf', fig)

        else:
            f1_acc, score = AU_metric(y_hat, y)
            self.log_dict({'val/f1': f1_acc[0], 'val/acc': f1_acc[1], 'val/score': score})

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        y_hat = self(batch)
        y_hat = y_hat.detach().cpu().numpy()

        if self.hparams.data_type == 'EXPR_Set':
            y_hat = np.argmax(y_hat, axis=-1)

        elif self.hparams.data_type == 'AU_Set':
            y_hat = (y_hat > 0.5).astype(int)

        else:
            y_hat = np.clip(y_hat, -0.99, 0.99)

        return y_hat

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get('learning_rate', 1e-3))
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.hparams.get('learning_rate', 3e-4),
                                    momentum=self.hparams.get('momentum', 0.9),
                                    weight_decay=self.hparams.get('weight_decay', 1e-4))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, self.hparams.get('gamma', 0.1))
        return [optimizer], [lr_scheduler]

    def configure_callbacks(self):
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor='val/score',
            mode='max',
            filename='epoch{epoch}-loss{val/loss:.2f}-score{val/score:.2f}',
            save_top_k=3,
            auto_insert_metric_name=False
        )
        return [checkpoint]

TYPE = ['VA_Set', 'EXPR_Set', 'AU_Set']


class MultiClassifier(pl.LightningModule):
    def __init__(self, params: dict):
        super().__init__()
        self.save_hyperparameters(params)

        self.backbone = torchvision.models.mobilenet_v2(pretrained=True)

        self.VA_head = ClassificationHead(input_dim=1000, target_dim=2)
        self.EXPR_head = ClassificationHead(input_dim=1000, target_dim=7)
        self.AU_head = ClassificationHead(input_dim=1000, target_dim=12)

        self.VA_loss = nn.MSELoss()
        self.EXPR_loss = nn.CrossEntropyLoss()
        self.AU_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        y_hat = [self.VA_head(embedding),
                 self.EXPR_head(embedding),
                 self.AU_head(embedding)]
        y_hat = torch.cat(y_hat, dim=1)

        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        va_loss = self.VA_loss(y_hat[:, :2], y[:, :2])
        expr_loss = self.EXPR_loss(y_hat[:, 2:9], y[:, 2].long())
        au_loss = self.AU_loss(y_hat[:, 9:], y[:, 3:].float())

        total_loss = va_loss + expr_loss + au_loss

        self.log_dict({'train/loss': total_loss,
                       'train/va_loss': va_loss,
                       'train/expr_loss': expr_loss,
                       'train/au_loss': au_loss}, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        va_loss = self.VA_loss(y_hat[:, :2], y[:, :2])
        expr_loss = self.EXPR_loss(y_hat[:, 2:9], y[:, 2].long())
        au_loss = self.AU_loss(y_hat[:, 9:], y[:, 3:].float())

        total_loss = va_loss + expr_loss + au_loss
        self.log_dict({'val/loss': total_loss,
                       'val/va_loss': va_loss,
                       'val/expr_loss': expr_loss,
                       'val/au_loss': au_loss}, on_epoch=True)

        y_hat = y_hat.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        item, sum = VA_metric(y_hat[:, :2], y[:, :2])
        self.log_dict({'val/CCC-V': item[0],
                       'val/CCC-A': item[1],
                       'val/va_score': sum}, on_epoch=True)

        f1_acc, score, _ = EXPR_metric(y_hat[:, 2:9], y[:, 2].astype(int))
        self.log_dict({'val/expr_f1': f1_acc[0],
                       'val/expr_acc': f1_acc[1],
                       'val/expr_score': score}, on_epoch=True)

        f1_acc, score = AU_metric(y_hat[:, 9:], y[:, 3:])
        self.log_dict({'val/au_f1': f1_acc[0],
                       'val/au_acc': f1_acc[1],
                       'val/au_score': score}, on_epoch=True)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get('learning_rate', 1e-3))
        # optimizer = torch.optim.SGD(self.parameters(),
        #                             lr=self.hparams.get('learning_rate', 3e-4),
        #                             momentum=self.hparams.get('momentum', 0.9),
        #                             weight_decay=self.hparams.get('weight_decay', 1e-4))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, self.hparams.get('gamma', 0.1))
        return [optimizer], [lr_scheduler]

    def configure_callbacks(self):
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor='val/loss',
            filename='epoch{epoch}-loss{val/loss:.2f}-va{val/va_score:.2f}-expr{val/expr_score:.2f}-au{val/au_score:.2f}',
            save_top_k=3,
            auto_insert_metric_name=False
        )
        return [checkpoint]

class Mono3DClassifier(MonoClassifier):
    def __init__(self, params: dict):
        super().__init__(params)
        self.save_hyperparameters(params)

        self.backbone = torchvision.models.mobilenet_v2(pretrained=True)
        self.backbone_3d = torchvision.models.mobilenet_v2(pretrained=True)

        if self.hparams.data_type == 'VA_Set':
            self.head = ClassificationHead(input_dim=2000, target_dim=2)
            self.loss = nn.MSELoss()
        elif self.hparams.data_type == 'EXPR_Set':
            self.head = ClassificationHead(input_dim=2000, target_dim=7)
            self.loss = nn.CrossEntropyLoss()
            # self.loss = lambda x, y: F.cross_entropy(x, y) + F.multi_margin_loss(x, y)
        else:
            self.head = ClassificationHead(input_dim=2000, target_dim=12)
            self.loss = nn.BCEWithLogitsLoss()
            # self.loss = lambda x, y: F.binary_cross_entropy_with_logits(x, y) + F.l1_loss(x, y)

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x[:, 0])
        embedding_3d = self.backbone_3d(x[:, 1])
        y_hat = self.head(torch.cat([embedding, embedding_3d], dim=-1))
        return y_hat

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class MonoPyramidClassifier(MonoClassifier):
    def __init__(self, params: dict):
        super().__init__(params)
        self.save_hyperparameters(params)

        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)

        if self.hparams.data_type == 'VA_Set':
            self.head = ClassificationHead(input_dim=1280, target_dim=2)
            self.loss = nn.MSELoss()
        elif self.hparams.data_type == 'EXPR_Set':
            self.head = ClassificationHead(input_dim=1280, target_dim=7)
            self.loss = nn.CrossEntropyLoss()
            # self.loss = lambda x, y: F.cross_entropy(x, y) + F.multi_margin_loss(x, y)
        else:
            self.head = ClassificationHead(input_dim=1280, target_dim=12)
            self.loss = nn.BCEWithLogitsLoss()
            # self.loss = lambda x, y: F.binary_cross_entropy_with_logits(x, y) + F.l1_loss(x, y)

    def forward(self, x):
        # use forward for inference/predictions
        layers = self.backbone(x)
        embedding = torch.cat([torch.mean(layer,dim=(-2,-1)) for layer in layers.values()], dim=-1)
        y_hat = self.head(embedding)
        return y_hat

class MultiPyramidClassifier(MultiClassifier):
    def __init__(self, params: dict):
        super().__init__(params)
        self.save_hyperparameters(params)

        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)

        self.VA_head = ClassificationHead(input_dim=1280, target_dim=2)
        self.EXPR_head = ClassificationHead(input_dim=1280, target_dim=7)
        self.AU_head = ClassificationHead(input_dim=1280, target_dim=12)

        self.VA_loss = nn.MSELoss()
        self.EXPR_loss = nn.CrossEntropyLoss()
        self.AU_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        # use forward for inference/predictions
        layers = self.backbone(x)
        embedding = torch.cat([torch.mean(layer, dim=(-2, -1)) for layer in layers.values()], dim=-1)
        y_hat = [self.VA_head(embedding),
                 self.EXPR_head(embedding),
                 self.AU_head(embedding)]
        y_hat = torch.cat(y_hat, dim=1)

        return y_hat

if __name__ == '__main__':
    model = MonoPyramidClassifier({'data_type':'VA_Set'})
    output = model.backbone(torch.rand(32, 3, 128, 128))
    print([(k, v.shape) for k, v in output.items()])
    ma = torch.cat([torch.mean(layer,dim=(-2,-1)) for layer in output.values()],dim=-1)
    print(ma.shape)
