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
parser.add_argument('--affectnet_pkl', type=str, default=r"/home/user1/dataset/AffectNet/annotation.pkl")
parser.add_argument('--save_path', type=str,
                    default=r"/home/user1/dataset/Aff-Wild/mixedAnnotation/mixed_VA_annotations.pkl")
args = parser.parse_args()
VA_list = ['valence', 'arousal']


def read_aff_wild2():
    total_data = pickle.load(open(args.aff_wild2_pkl, 'rb'))
    # training set
    train_data = total_data['VA_Set']['Train_Set']
    paths = []
    labels = []
    for video in train_data.keys():
        data = train_data[video]
        labels.append(np.stack([data['valence'], data['arousal']], axis=1))
        paths.append(data['path'].values)
    paths = np.concatenate(paths, axis=0)
    labels = np.concatenate(labels, axis=0)
    train_data = {'label': labels, 'path': paths}
    # validation set
    val_data = total_data['VA_Set']['Validation_Set']
    paths = []
    labels = []
    for video in val_data.keys():
        data = val_data[video]
        labels.append(np.stack([data['valence'], data['arousal']], axis=1))
        paths.append(data['path'].values)
    paths = np.concatenate(paths, axis=0)
    labels = np.concatenate(labels, axis=0)
    val_data = {'label': labels, 'path': paths}
    return train_data, val_data


def merge_two_datasets():
    data_aff_wild2, data_aff_wild2_val = read_aff_wild2()
    # downsample x 5 the training set in aff_wild training set
    aff_wild_train_labels = data_aff_wild2['label']
    aff_wild_train_paths = data_aff_wild2['path']
    length = len(aff_wild_train_labels)
    index = [True if i % 5 == 0 else False for i in range(length)]
    aff_wild_train_labels = aff_wild_train_labels[index]
    aff_wild_train_paths = aff_wild_train_paths[index]
    data_aff_wild2 = {'label': aff_wild_train_labels, 'path': aff_wild_train_paths}

    data_affectnet = pickle.load(open(args.affectnet_pkl, 'rb'))
    labels = []
    paths = []
    labels.append(np.stack([data_affectnet['valence'], data_affectnet['arousal']], axis=1))
    paths.append(data_affectnet['path'])
    paths = np.concatenate(paths, axis=0)
    labels = np.concatenate(labels, axis=0)
    data_affectnet = {'label': labels, 'path': paths}
    data_merged = {'label': np.concatenate((data_aff_wild2['label'], data_affectnet['label']), axis=0),
                   'path': list(data_aff_wild2['path']) + list(data_affectnet['path'])}
    print("Aff-wild2 :{}".format(len(data_aff_wild2['label'])))
    print("AffectNet:{}".format(len(data_affectnet['label'])))
    print("MergedData:{}".format(len(data_merged['label'])))
    return {'Train_Set': data_merged, 'Validation_Set': data_aff_wild2_val}


def plot_distribution(data):
    all_samples = data['label']
    plt.hist2d(all_samples[:, 0], all_samples[:, 1], bins=(20, 20), cmap=plt.cm.jet)
    plt.xlabel("Valence")
    plt.ylabel('Arousal')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    data_file = merge_two_datasets()
    pickle.dump(data_file, open(args.save_path, 'wb'))
    plot_distribution(data_file['Train_Set'])
