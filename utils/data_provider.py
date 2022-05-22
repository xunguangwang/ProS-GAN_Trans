import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data.sampler as sampler
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import torchvision
from torchvision import transforms


class NUS_WIDE(Dataset):
    def __init__(self,
                 data_path,
                 img_filename,
                 label_filename,
                 transform=transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     transforms.Normalize([0.462, 0.445, 0.415],
                                          [0.274, 0.259, 0.287])
                 ])):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)


class NUS_WIDE_G(Dataset):
    def __init__(self,
                 data_path,
                 img_filename,
                 label_filename,
                 transform=transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ])):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)
        # fp_label = open(label_filepath, 'r')
        # labels = [int(x.strip()) for x in fp_label]
        # fp_label.close()
        # self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index]).float()
        # label = torch.LongTensor([self.label[index]])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)


class HashingDataset(Dataset):
    def __init__(self,
                 data_path,
                 img_filename,
                 label_filename,
                 transform=transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor()
                 ])):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index]).float()
        return img, label, index

    def __len__(self):
        return len(self.img_filename)


def get_mean_std(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=10)
    s_mean = np.array([0, 0, 0], dtype=float)
    s_std = np.array([0, 0, 0], dtype=float)
    c = 0
    for data in dataloader:
        batch_image = data[0]
        mean = np.mean(batch_image.numpy(), axis=(0, 2, 3))
        std = np.std(batch_image.numpy(), axis=(0, 2, 3))
        s_mean += mean
        s_std += std
        c += 1
    return s_mean / c, s_std / c


class SubsetSampler(sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def load_label(filename, data_dir):
    label_filepath = os.path.join(data_dir, filename)
    label = np.loadtxt(label_filepath, dtype=np.int64)
    return torch.from_numpy(label).float()