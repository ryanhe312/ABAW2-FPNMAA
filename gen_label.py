from argparse import ArgumentParser
import pytorch_lightning as pl
import pretty_errors
import yaml
import numpy as np
from os.path import *

from datasets import *
from model import *

def cli_main():
    pl.seed_everything(1234, workers=True)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # model
    # ------------
    model_args = yaml.safe_load(open(args.config, 'r')) if args.config else {}
    dm = AffWild2DataModule(model_args)
    model = MonoPyramidClassifier.load_from_checkpoint(args.checkpoint, strict=False) if args.checkpoint else MonoPyramidClassifier(model_args)

    # ------------
    # predicting
    # ------------
    pred = trainer.predict(model, datamodule=dm)

    # ------------
    # writing
    # ------------
    soft_label_dict = dict(zip(dm.predict_dataset.image, np.concatenate(pred, axis=0)))
    hard_label_dict = dict(zip(dm.train_dataset.image, dm.train_dataset.label))
    label_dict = soft_label_dict
    label_dict.update(hard_label_dict)

    file = open(join(dm.dataset_dir, 'file.txt'), 'w+')
    file.writelines([join(basename(dirname(path)), basename(path)) + '\n' for path in label_dict.keys()])
    file.close()

    label = open(join(dm.dataset_dir, dm.data_type+'.txt'), 'w+')
    label.write(dm.data_type+'\n')
    label.writelines([','.join(label.astype(str))+'\n' for label in label_dict.values()])
    label.close()


if __name__ == '__main__':
    cli_main()
