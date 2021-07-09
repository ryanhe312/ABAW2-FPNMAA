from argparse import ArgumentParser
import pytorch_lightning as pl
import pretty_errors
import yaml

from datasets import *
from model import *

def cli_main():
    pl.seed_everything(1234, workers=True)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--train', action="store_true")
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
    dm = BalancedDataModule(model_args)
    model = MonoPyramidClassifier.load_from_checkpoint(args.checkpoint, strict=False) if args.checkpoint else MonoPyramidClassifier(model_args)

    # ------------
    # training
    # ------------
    if args.train: trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)


if __name__ == '__main__':
    cli_main()
