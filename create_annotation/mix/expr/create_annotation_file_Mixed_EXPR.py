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
parser.add_argument('--expw_pkl', type=str, default=r"/home/user1/dataset/ExpW/annotations.pkl")

parser.add_argument('--save_path', type=str,
                    default=r"/home/user1/dataset/Aff-Wild/mixedAnnotation/mixed_EXPR_annotations.pkl")
parser.add_argument('--fig_path', type=str,
                    default=r"/home/user1/dataset/Aff-Wild/mixedAnnotation/mixed_EXPR_trainset_distribution.png")
args = parser.parse_args()
Expr_list = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']


def read_aff_wild2():
    total_data = pickle.load(open(args.aff_wild2_pkl, 'rb'))
    # train set
    data = total_data['EXPR_Set']['Train_Set']
    paths = []
    labels = []
    for video in data.keys():
        df = data[video]
        labels.append(df['label'].values.astype(np.float32))
        paths.append(df['path'].values)
    paths = np.concatenate(paths, axis=0)
    labels = np.concatenate(labels, axis=0)
    data = {'label': labels, 'path': paths}
    # validation set
    val_data = total_data['EXPR_Set']['Validation_Set']
    paths = []
    labels = []
    for video in val_data.keys():
        df = val_data[video]
        labels.append(df['label'].values.astype(np.float32))
        paths.append(df['path'].values)
    paths = np.concatenate(paths, axis=0)
    labels = np.concatenate(labels, axis=0)
    val_data = {'label': labels, 'path': paths}
    return data, val_data


def merge_datasets():
    data_aff_wild2, data_aff_wild2_val = read_aff_wild2()
    data_ExpW = pickle.load(open(args.expw_pkl, 'rb'))
    data_affectnet = pickle.load(open(args.affectnet_pkl, 'rb'))
    # change the label integer, because of the different labelling in two datasets
    data_merged = {
        'label': np.concatenate((data_aff_wild2['label'], data_ExpW['label'], data_affectnet['expr']), axis=0),
        'path': list(data_aff_wild2['path']) + data_ExpW['path'] + list(data_affectnet['path'])}
    print("Dataset\t" + "\t".join(Expr_list))
    print("Aff_wild2 dataset:\t" + "\t".join([str(sum(data_aff_wild2['label'] == i)) for i in range(len(Expr_list))]))
    print("ExpW dataset:\t" + "\t".join([str(sum(data_ExpW['label'] == i)) for i in range(len(Expr_list))]))
    print("AffectNet dataset:\t" + "\t".join([str(sum(data_affectnet['expr'] == i)) for i in range(len(Expr_list))]))
    # ====================downsample===================================
    # undersample the neutral samples and happy samples by 10
    # undersample the sad samples by 3

    # neutral
    # labels = data_merged['label']
    # is_neutral = labels == 0
    # keep_10 = np.array([True if index % 10 == 0 else False for index in range(len(labels))])
    # to_drop = is_neutral & ~keep_10
    # labels = data_merged['label'][~to_drop]
    # paths = np.array(data_merged['path'])[~to_drop]
    # data_merged.update({'label': labels, 'path': paths})

    # happy
    # labels = data_merged['label']
    # paths = np.array(data_merged['path'])
    # is_happy = labels == 4
    # keep_10 = np.array([True if index % 10 == 0 else False for index in range(len(labels))])
    # to_drop = is_happy & ~keep_10
    # data_merged.update({'label': labels[~to_drop], 'path': paths[~to_drop]})

    # sad
    # labels = data_merged['label']
    # paths = data_merged['path']
    # is_sad = labels == 5
    # keep_4 = np.array([True if index % 4 == 0 else False for index in range(len(labels))])
    # to_drop = is_sad & ~keep_4
    # data_merged.update({'label': labels[~to_drop], 'path': paths[~to_drop]})

    return {'Train_Set': data_merged, 'Validation_Set': data_aff_wild2_val}


def plot_distribution(data):
    all_samples = data['label']
    histogram = np.zeros(len(Expr_list))
    for i in range(len(Expr_list)):
        find_true = sum(all_samples == i)
        histogram[i] = find_true
    print(Expr_list)
    print(histogram)
    histogram = histogram / all_samples.shape[0]
    plt.bar(np.arange(len(Expr_list)), histogram)
    plt.xticks(np.arange(len(Expr_list)), Expr_list)
    plt.savefig(args.fig_path)
    plt.show()


if __name__ == '__main__':
    data_file = merge_datasets()
    pickle.dump(data_file, open(args.save_path, 'wb'))
    plot_distribution(data_file['Train_Set'])
