import pickle
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt
import matplotlib
import glob
import pandas as pd
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description='read two annotations files')
parser.add_argument('--aff_wild2_pkl', type=str,
                    default='/home/user1/dataset/Aff-Wild/annotations/annotations.pkl')
parser.add_argument('--save_path', type=str,
                    default=r"/home/user1/dataset/Aff-Wild/mixedAnnotation/mixed_AU_annotations.pkl")
parser.add_argument('--fig_path', type=str,
                    default=r"/home/user1/dataset/Aff-Wild/mixedAnnotation/mixed_AU_trainset_distribution.png")
args = parser.parse_args()
AU_list = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']


def read_aff_wild2():
    total_data = pickle.load(open(args.aff_wild2_pkl, 'rb'))
    # train set
    data = total_data['AU_Set']['Train_Set']
    paths = []
    labels = []
    for video in data.keys():
        df = data[video]
        label = []
        for au in AU_list:
            label.append(df[au].values.astype(np.float32))
        labels.append(np.stack(label, axis=1))
        paths.append(df['path'].values)
    paths = np.concatenate(paths, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(labels.shape)
    data = {'label': labels, 'path': paths}
    # validation set
    val_data = total_data['AU_Set']['Validation_Set']
    paths = []
    labels = []
    for video in val_data.keys():
        df = val_data[video]
        label = []
        for au in AU_list:
            label.append(df[au].values.astype(np.float32))
        labels.append(np.stack(label, axis=1))
        paths.append(df['path'].values)
    paths = np.concatenate(paths, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(labels.shape)
    val_data = {'label': labels, 'path': paths}
    return data, val_data


def merge_datasets():
    data_aff_wild2, data_aff_wild2_val = read_aff_wild2()
    # change the label integer, because of the different labelling in two datasets
    return {'Train_Set': data_aff_wild2, 'Validation_Set': data_aff_wild2_val}


if __name__ == '__main__':
    data_file = merge_datasets()
    pickle.dump(data_file, open(args.save_path, 'wb+'))
