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
parser.add_argument('--csv_file_dir', type=str, default=r"/home/user1/dataset/AffectNet/Manually_Annotated_file_lists")
parser.add_argument('--img_dir', type=str, default=r"/home/user1/dataset/AffectNet/Manually_Annotated_Images")
parser.add_argument('--distribution_output', type=str, default=r"/home/user1/dataset/AffectNet.distribution.jpg")
parser.add_argument('--save_path', type=str, default=r"/home/user1/dataset/AffectNet/annotation.pkl")
args = parser.parse_args()
expr_list = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
expr_transform = [0, 4, 5, 6, 3, 2, 1]
'''
expr_category  affectnet aff-wild2
  neutral          0         0
  anger            6         1
  disgust          5         2
  fear             4         3
  happy            1         4
  sad              2         5
  surprise         3         6
'''


def read_csv():
    files = os.listdir(args.csv_file_dir)
    dire = []
    expr = []
    valence = []
    arousal = []
    dict = {}
    for file in tqdm(files):
        obj = pd.read_csv(os.path.join(args.csv_file_dir, file))
        # 过滤数据
        obj = obj[
            obj["expression"].between(0, 6) & obj["valence"].between(-1.0, 1.0) & obj["arousal"].between(-1.0, 1.0)]
        dire.extend(obj["subDirectory_filePath"].values)
        expr.extend(obj["expression"].values)
        valence.extend(obj["valence"].values)
        arousal.extend(obj["arousal"].values)

    # 标签转换
    for i in range(len(expr)):
        label = int(expr[i])
        expr[i] = expr_transform[label]

    # update path
    prefix = '/'.join(args.img_dir.split("/"))
    path = [prefix + '/' + subdire for subdire in dire]

    # update dict
    dict["path"] = path
    dict["expr"] = np.array(expr)
    # dtype=int64
    dict["valence"] = np.array(valence)
    # dtype=float64
    dict["arousal"] = np.array(arousal)
    # dtype=float64

    # save file
    df = pd.DataFrame.from_dict(dict)
    pickle.dump(df, open(args.save_path, 'wb'))
    return df


def plot_distribution(data_file):
    histogram = np.zeros(len(expr_list))
    all_samples = data_file['expr']
    for i in range(7):
        find_true = sum(all_samples == i)
        histogram[i] = find_true / all_samples.shape[0]
    plt.bar(np.arange(len(expr_list)), histogram)
    plt.xticks(np.arange(len(expr_list)), expr_list)
    plt.savefig(args.distribution_output)
    plt.show()


def print_distribution(data_file):
    histogram = np.zeros(len(expr_list))
    all_samples = data_file['expr']
    for i in range(7):
        find_true = sum(all_samples == i)
        histogram[i] = find_true
    print(expr_list)
    print(histogram)


if __name__ == "__main__":
    data_file = read_csv()
    # data_file = pickle.load(open(args.save_path, 'rb'))
    plot_distribution(data_file)
    print_distribution(data_file)
    print(len(data_file))
