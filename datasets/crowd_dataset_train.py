import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import h5py
import cv2
import glob
import skimage.transform
from time import sleep


def get_train_and_validation_indices(num, ratio):
    val_count = int(num * ratio)
    random.seed(42)
    all_indices = list(range(1, num + 1))
    val_indices = random.sample(all_indices, val_count)
    train_indices = [item for item in all_indices if item not in val_indices]
    return np.array(train_indices, dtype=int), np.array(val_indices, dtype=int)


class CrowdDataset_Train(Dataset):
    def __init__(self, root_path, split, transform=None):
        # get all images to be tested
        root = glob.glob(os.path.join(root_path, 'train_data/images/*.jpg'))

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.split = split
        val_ratio = 0.2
        train_indices, val_indices = get_train_and_validation_indices(self.nSamples, val_ratio)
        if self.split == 'train':
            self.indices = np.array(train_indices)
        else:  # val
            self.indices = np.array(val_indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # get the image path
        img_path = self.lines[index]

        img, target = load_data(img_path)
        # perform data augumentation
        img = img.resize((300, 300), resample=Image.NEAREST)
        if self.transform is not None:
            img = self.transform(img)

        img = torch.Tensor(img)

        resized_density_map = skimage.transform.resize(target, (300, 300))
        resized_density_map *= np.sum(target) / np.sum(resized_density_map)
        resized_density_map = torch.tensor(resized_density_map)
        resized_density_map = resized_density_map.float()
        resized_density_map = torch.unsqueeze(resized_density_map, 0)

        return img, resized_density_map


def load_data(img_path):
    # get the path of the ground truth
    gt_path = img_path.replace('.jpg', '_sigma4.h5').replace('images', 'ground_truth')
    # open the image
    img = Image.open(img_path).convert('RGB')
    # load the ground truth
    while True:
        try:
            gt_file = h5py.File(gt_path)
            break
        except:
            sleep(2)
    target = np.asarray(gt_file['density'])

    return img, target