import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms as tt
from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import dataloaders.chexpert as chex
import dataloaders.kaggle_rsna as kaggle_rsna

class MergedDataset():
    chestXray_train_ds, chestXray_test_ds = [], []
    CheX_train_ds, CheX_test_ds = [], []
    kaggleRsna_train_ds, kaggleRsna_test_ds = [], []

    positiveClassSize_train = 0
    negativeClassSize_train = 0
    positiveClassSize_test = 0
    negativeClassSize_test = 0

    train_ds = None
    test_ds = None
    train_dl = None
    test_dl = None

    def __init__(self, device, transformLoadingChest=None, transformLoadingCheX=None, transformLoadingRsna=None, transformTrain=None, transformTest=None, chest_xray=False, cheX=False, kaggle_rsna=False, batch_size=64, train_percentage=0.8, kaggleRsna_drop_normal_percentage=0.50, split_seed=2024):
        seed = split_seed
        torch.manual_seed(seed)

        # Load datasets
        if chest_xray:
            self.chestXray = self.__loadChestXray(transformLoadingChest)

            if train_percentage == 0:
                self.chestXray_train_ds, self.chestXray_test_ds = random_split(self.chestXray, [0, len(self.chestXray)])
            elif train_percentage == 1:
                self.chestXray_train_ds, self.chestXray_test_ds = random_split(self.chestXray, [len(self.chestXray), 0])
            else:
                chestXray_train_size = round(len(self.chestXray) * train_percentage)
                chestXray_test_size = len(self.chestXray) - chestXray_train_size
                self.chestXray_train_ds, self.chestXray_test_ds = random_split(self.chestXray, [chestXray_train_size, chestXray_test_size])
            temp = torch.utils.data.ConcatDataset([self.chestXray])
            res = self.__findClassSizes(self.chestXray_train_ds, temp)
            if len(res) == 2:
                self.positiveClassSize_train += res[1]
                self.negativeClassSize_train += res[0]
            res = self.__findClassSizes(self.chestXray_test_ds, temp)
            if len(res) == 2:
                self.positiveClassSize_test += res[1]
                self.negativeClassSize_test += res[0]

        if cheX:
            self.CheX = self.__loadCheX(transformLoadingCheX)

            if train_percentage == 0:
                self.CheX_train_ds,  self.CheX_test_ds = random_split(self.CheX, [0, len(self.CheX)])
            elif train_percentage == 1:
                self.CheX_train_ds, self.CheX_test_ds = random_split(self.CheX, [len(self.CheX), 0])
            else:
                CheX_train_size = round(len(self.CheX) * train_percentage)
                CheX_test_size = len(self.CheX) - CheX_train_size
                self.CheX_train_ds, self.CheX_test_ds = random_split(self.CheX, [CheX_train_size, CheX_test_size])
            temp = torch.utils.data.ConcatDataset([self.CheX])
            res = self.__findClassSizes(self.CheX_train_ds, temp)
            if len(res) == 2:
                self.positiveClassSize_train += res[1]
                self.negativeClassSize_train += res[0]
            res = self.__findClassSizes(self.CheX_test_ds, temp)
            if len(res) == 2:
                self.positiveClassSize_test += res[1]
                self.negativeClassSize_test += res[0]

        if kaggle_rsna:
            self.kaggleRsna = self.__loadKaggleRsna(transformLoadingRsna, drop_percentage=kaggleRsna_drop_normal_percentage)

            if train_percentage == 0:
                self.kaggleRsna_train_ds, self.kaggleRsna_test_ds = random_split(self.kaggleRsna, [0, len(self.kaggleRsna)])
            elif train_percentage == 1:
                self.kaggleRsna_train_ds, self.kaggleRsna_test_ds = random_split(self.kaggleRsna, [len(self.kaggleRsna), 0])
            else:
                kaggleRsna_train_size = round(len(self.kaggleRsna) * train_percentage)
                kaggleRsna_test_size = len(self.kaggleRsna) - kaggleRsna_train_size
                self.kaggleRsna_train_ds, self.kaggleRsna_test_ds = random_split(self.kaggleRsna, [kaggleRsna_train_size, kaggleRsna_test_size])
            temp = torch.utils.data.ConcatDataset([self.kaggleRsna])
            res = self.__findClassSizes(self.kaggleRsna_train_ds, temp)
            if len(res) == 2:
                self.positiveClassSize_train += res[1]
                self.negativeClassSize_train += res[0]
            res = self.__findClassSizes(self.kaggleRsna_test_ds, temp)
            if len(res) == 2:
                self.positiveClassSize_test += res[1]
                self.negativeClassSize_test += res[0]

        # Merge the training and testing datasets
        self.train_ds = torch.utils.data.ConcatDataset([self.chestXray_train_ds, self.CheX_train_ds, self.kaggleRsna_train_ds])
        self.test_ds = torch.utils.data.ConcatDataset([self.chestXray_test_ds, self.CheX_test_ds, self.kaggleRsna_test_ds])

        if len(self.train_ds) > 0:
            train_ds = TransformedDataset(self.train_ds, transformTrain)
            train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
            self.train_dl = DeviceDataLoader(train_dl, device)

        if len(self.test_ds) > 0:
            test_ds = TransformedDataset(self.test_ds, transformTest)
            test_dl = DataLoader(test_ds, batch_size, num_workers=4, pin_memory=False)
            self.test_dl = DeviceDataLoader(test_dl, device)

    def getDataLoader(self):
        return self.train_dl, self.test_dl
        
    def getTrainClasses(self):
        return self.negativeClassSize_train, self.positiveClassSize_train

    def getTestClasses(self):
        return self.negativeClassSize_test, self.positiveClassSize_test

    def __findClassSizes(self, dataset, total):
        if dataset == 0:
            return {1: 0, 0: 0}

        class_count = {}
        indices = dataset.indices

        for global_idx in indices:
            for j, offset in enumerate(total.cumulative_sizes):
                if global_idx < offset:
                    local_idx = global_idx if j == 0 else global_idx - total.cumulative_sizes[j - 1]
                    subset = total.datasets[j]
                    break
            label = subset.targets[local_idx]
            if label not in class_count:
                class_count[label] = 0
            class_count[label] += 1
        return class_count

    def getSize(self):
        return len(self.train_ds) + len(self.test_ds)

    def __loadChestXray(self, transformLoading):
        return datasets.ImageFolder('../../bigdata/chest_xray-3', transform=transformLoading)

    def __loadCheX(self, transformLoading):
        dataset = chex.CheXDataset('../../bigdata/CheXpert-v1.0-small', [transformLoading])
        return dataset
    
    def __loadKaggleRsna(self, transformLoading, drop_percentage=0.50):
        dataset = kaggle_rsna.RSNADataset('../../bigdata/kaggle-rsna', [transformLoading])
        dataset.drop(0, drop_percentage, 42)
        return dataset

class TransformedDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device) # yield will stop here, perform other steps, and the resumes to the next loop/batch

    def __len__(self):
        """Number of batches"""
        return len(self.dl) 