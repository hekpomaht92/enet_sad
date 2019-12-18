import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
from random import shuffle
from config import Config

cfg = Config()


class ImageGenerator(Dataset):
    '''Converte images to tensors'''
    def __init__(self, input_path, num_images):
        with open(input_path, 'r') as f:
            self.img_list = f.readlines()
        self.num_images = num_images
        self.transforms_color = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)])

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = self.transforms_color(Image.open(img_path_list.split()[0]))
        label = np.array(Image.open(img_path_list.split()[-1]))
        label = torch.squeeze(torch.tensor(label))
        sample = {'data': data, 'label': label}
        return sample


class ImageGeneratorTest(Dataset):
    '''Converte images to tensors'''
    def __init__(self, input_path, num_images):
        with open(input_path, 'r') as f:
            self.img_list = f.readlines()
        self.num_images = num_images
        self.transforms_color = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)])

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = self.transforms_color(Image.open(img_path_list.split()[0]))
        raw_image = cv2.imread(img_path_list.split()[0])
        label = np.array(Image.open(img_path_list.split()[-1]))
        label = torch.squeeze(torch.tensor(label))
        sample = {'data': data, 'label': label, 'raw_image': raw_image}
        return sample


def computing_mean_std_weightcoef():
    '''Compute mean and std of train dataset'''
    _mean_absolut = np.array([0.0, 0.0, 0.0])
    _std_absolut = np.array([0.0, 0.0, 0.0])
    _weight_coef = 0.

    print('Start: {}'.format(cfg.train_dir))
    with open(cfg.train_dir, 'r') as f:
        img_list = f.readlines()

    for line in img_list:
        line = line.split()
        x_data = cv2.imread(line[0])
        x_data = x_data.astype("float32")
        x_data /= 255
        means = x_data.mean(axis=(0, 1), dtype='float64')
        stds = x_data.std(axis=(0, 1), dtype='float64')

        _mean_absolut += means
        _std_absolut += stds

        y_data = cv2.imread(line[1])[...,0]
        _weight_coef += np.sum(y_data) / (y_data.shape[0] * y_data.shape[1])
        print(line)

    _mean_absolut /= cfg.num_train_images
    _std_absolut /= cfg.num_train_images
    _weight_coef /= cfg.num_train_images
    _mean_absolut = _mean_absolut[::-1]
    _std_absolut = _std_absolut[::-1]

    with open('mean_std.txt', 'w') as f:
        f.write('mean: {}\nstd: {}\nweight_coef: {}'
                .format(_mean_absolut, _std_absolut, _weight_coef))

def create_train_val_test_lists():
    '''Split train, val and test files from merged file of all images'''
    merged_list_file_path = os.path.join('lists', 'merged_file.txt')
    img_list = []

    with open(merged_list_file_path, 'r') as f:
        while True:
            lines = f.readline()
            if not lines:
                break
            item = lines.strip().split()
            img_list.append(item)    
    shuffle(img_list)

    for i in range(len(img_list)):
        if i <= int(len(img_list)*0.7):
            with open(os.path.join('lists', 'train.txt'), 'a') as f:
                f.write(' '.join(img_list[i]) + '\n')
        elif i <= int(len(img_list)*0.9):
            with open(os.path.join('lists', 'val.txt'), 'a') as f:
                f.write(' '.join(img_list[i]) + '\n')
        else:
            with open(os.path.join('lists', 'test.txt'), 'a') as f:
                f.write(' '.join(img_list[i]) + '\n')

def create_list_for_video():
    '''Sort merged_file to video_list'''
    merged_list_file_path = os.path.join('lists', 'merged_file.txt')
    img_list = []

    with open(merged_list_file_path, 'r') as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            if line.split()[0].split('/')[8] == 'Camera_5':
                img_list.append(line)    
    img_list.sort()

    with open(os.path.join('lists', 'video_list.txt'), 'a') as f:
        for line in img_list:
            f.write(line + '\n')

if __name__ == '__main__':
    # create_train_val_test_lists()
    # computing_mean_std_weightcoef()
    create_list_for_video()
    print('Complite')
