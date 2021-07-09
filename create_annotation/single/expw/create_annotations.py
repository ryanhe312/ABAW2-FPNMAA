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

parser = argparse.ArgumentParser(description='save annotations')
parser.add_argument('--vis', action='store_true',
                    help='whether to visualize the distribution')
parser.add_argument('--image_dir', type=str, default=r"/home/user1/dataset/ExpW/origin",
                    help='image dir')
parser.add_argument('--lst_file', type=str, default=r"/home/user1/dataset/ExpW/label.lst")
parser.add_argument('--output_dir', type=str, default=r"/home/user1/dataset/ExpW/images")

parser.add_argument('--save_path', type=str, default=r"/home/user1/dataset/ExpW/annotations.pkl")
parser.add_argument('--distribution_output', type=str, default=r"/home/user1/dataset/ExpW/distribution.jpg")
args = parser.parse_args()
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
expr_list = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
expr_transform = [1, 2, 3, 4, 5, 6, 0]
'''
expr_category    ExpW aff-wild2
  neutral          6         0
  anger            0         1
  disgust          1         2
  fear             2         3
  happy            3         4
  sad              4         5
  surprise         5         6
'''


def read_lst(lst_file):
    with open(lst_file, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    data = {'name': [], 'face_id': [], 'ymin': [], 'xmin': [], 'xmax': [], 'ymax': [], 'confidence': [], 'emotion': []}
    for l in lines:
        l = l.split(" ")
        data['name'].append(l[0])
        data['face_id'].append(int(l[1]))
        data['ymin'].append(int(l[2]))
        data['xmin'].append(int(l[3]))
        data['xmax'].append(int(l[4]))
        data['ymax'].append(int(l[5]))
        data['confidence'].append(float(l[6]))
        data['emotion'].append(expr_transform[int(l[7])])
    df = pd.DataFrame.from_dict(data)
    return df


def read_all_image():
    data_file = {}
    df = read_lst(args.lst_file)
    paths = []
    labels = []
    for i in tqdm(range(len(df)), total=len(df)):
        line = df.iloc[i]
        # iloc函数，通过行号取对应行数据
        path = os.path.join(args.image_dir, line['name'])
        if os.path.exists(path):
            bbox = line[['xmin', 'ymin', 'xmax', 'ymax']].values
            img = Image.open(path).convert("RGB")
            face = img.crop(tuple(bbox))
            save_path = os.path.join(args.output_dir, line['name'])
            save_path = '/'.join(save_path.split('/'))
            face.save(save_path)
            paths.append(save_path)
            labels.append(line['emotion'])
    data_file['label'] = np.array(labels)
    data_file['path'] = paths
    pickle.dump(data_file, open(args.save_path, 'wb'))
    return data_file


def plot_distribution(data_file):
    histogram = np.zeros(len(expr_list))
    all_samples = data_file['label']
    for i in range(7):
        find_true = sum(all_samples == i)
        histogram[i] = find_true / all_samples.shape[0]
    plt.bar(np.arange(len(expr_list)), histogram)
    plt.xticks(np.arange(len(expr_list)), expr_list)
    plt.savefig(args.distribution_output)
    plt.show()


def print_distribution(data_file):
    histogram = np.zeros(len(expr_list))
    all_samples = data_file['label']
    for i in range(7):
        find_true = sum(all_samples == i)
        histogram[i] = find_true
    print(expr_list)
    print(histogram)


if __name__ == '__main__':
    data_file = read_all_image()
    # data_file = pickle.load(open(args.save_path, 'rb'))
    plot_distribution(data_file)
    print_distribution(data_file)
