# ABAW2-FPNMAA

We participated in ICCV 2021: 2nd Workshop and Competition on Affective Behavior Analysis in-the-wild (ABAW). And more details can be found in [our paper](http://arxiv.org/abs/2107.03670).

## Requirement

We build the model on Pytorch 1.7.1 and use Pytorch Lightning framework. 

```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch
pip install pytorch-lightning seaborn pretty_errors pandas PyYAML scikit-learn
```

## Testing

### download the dataset and create annotations

Please refer to the [official website](https://ibug.doc.ic.ac.uk/resources/iccv-2021-2nd-abaw/) of ABAW for Aff-wild2 dataset. 
And [this link](http://mohammadmahoor.com/affectnet/) for AffectNet,
[this link](http://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html) for ExpW. The final directory tree will be like this.

```
dataset/
├── AffectNet
│   ├── Manually_Annotated_file_lists
│   └── Manually_Annotated_Images
├── ExpW
│   ├── label.lst
│   ├── origin
│   └── readme.txt
├── Aff-Wild
│   ├── annotations
│   ├── cropped_aligned
│   ├── mixedAnnotation
```

1. open `create_annotations.py` files in separate dataset folders under `create_annotation/single/` and change the path to dataset there. 
2. run each `create_annotations.py` to get annotation for each dataset.
3. open `create_annotation_file_Mixed_*.py` files in separate task folders under `create_annotation/mix/` and change the path to dataset there.
4. run each `create_annotation_file_Mixed_*.py` to get annotation for each task.

### download the model and generate single task model

Please download the multi-task model from [google drive](https://drive.google.com/file/d/1tUpdqS_Reu4oNaqBBOr_XKPEuV9oofLi/view?usp=sharing).

1. fill in the model name in the `ckpt_process.py`.
2. put it in the folder where the model is.
3. run `python ckpt_process.py`.
4. you will get `multi_va.ckpt`, `multi_expr.ckpt` and `multi_au.ckpt`.

### run test script

1. edit the `dataset_dir` in configuration files in `configs/`.
2. run test scripts. take va prediction for example.

```shell
python mono_fit.py --gpus 1 --config configs/train_va.yml --checkpoint /path/to/multi_va.ckpt 
```

## Training

### train single task model

take va training for example.

```shell
python mono_fit.py --gpus 1 --config configs/train_va.yml --train --max_epochs 20 --limit_train_batch 0.25 
```

### generate soft label

take va label generation for example.

```shell
python gen_label.py  --gpus 1 --config configs/train_va.yml --checkpoint /path/to/single_va.ckpt
```

### train multi task model

```shell
python multi_fit.py --gpus 1 --config configs/train_multi.yml --train --max_epochs 20 
```


## Citation

If your work or research benefits from this repo, please cite the paper below.

```
@misc{he2021feature,
      title={Feature Pyramid Network for Multi-task Affective Analysis}, 
      author={Ruian He and Zhen Xing and Bo Yan},
      year={2021},
      eprint={2107.03670},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
