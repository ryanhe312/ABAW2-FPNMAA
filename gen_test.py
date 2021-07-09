import os.path
from argparse import ArgumentParser
import pytorch_lightning as pl
import pretty_errors
import yaml
import numpy as np
from os.path import *

from datasets import *
from model import *

CLASS = {
    'VA_Set': 2,
    'EXPR_Set': 1,
    'AU_Set': 12
}

TYPE = {
    'VA_Set': float,
    'EXPR_Set': int,
    'AU_Set': int
}

TEXT = {
    'VA_Set': 'valence,arousal\n',
    'EXPR_Set': 'Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise\n',
    'AU_Set': 'AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26\n'
}

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
    label_dict = dict(zip(dm.predict_dataset.image, np.concatenate(pred, axis=0)))
    result_dict = {
        video :
        {
            i : -np.ones(CLASS[dm.data_type],dtype=TYPE[dm.data_type])

            for i in range(dm.predict_dataset.test_set[video])
        }
        for video in dm.predict_dataset.test_set.keys()
    }

    for name in dm.predict_dataset.image:
        video = os.path.basename(os.path.dirname(name))
        frame = int(os.path.basename(name)[:-4])-1
        result_dict[video][frame] = label_dict[name]

    for video in dm.predict_dataset.test_set.keys():
        for frame in range(dm.predict_dataset.test_set[video]):
            if frame and np.all(result_dict[video][frame] == -1):
                result_dict[video][frame] = result_dict[video][frame-1]

    output_dir = join(dm.dataset_dir, 'test_result', dm.data_type)
    os.makedirs(output_dir,exist_ok=True)
    for video in dm.predict_dataset.test_set.keys():
        label = open(join(output_dir, f'{video}.txt'), 'w+')
        label.write(TEXT[dm.data_type])
        label.writelines([','.join(label.astype(str))+'\n' for label in result_dict[video].values()])
        label.close()


if __name__ == '__main__':
    cli_main()
