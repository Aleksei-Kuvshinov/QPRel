# -*- coding: utf-8 -*-

import torchvision

import torch
import torch.utils.data as tud

from rdl import PATH_DATA

import logging




class Dataset_(tud.Dataset):
    """ Class for a general dataset generated from given samples X and target y.

    It allows you to load the samples in batches by iterating through Dataset.loader and use the functionality of torch.data.DataLoader. However, you need the whole dataset to be loaded into the memory to generate an object of this class, so it isn't suitable for very large instances.

    Attributes:
    -----------
    X : torch.Tensor or numpy.ndarray
        2d matrix containing the samples (#samples x #features.)
    y : torch.Tensor or numpy.ndarray
        Vector containing the labels with length #samples.
    loader : torch.utils.data.DataLoader
        Loads batches of data with Dataset.loader.batch_size many samples.

    """


    def __init__(self, X, y, classes=None,
        name="OTHER", reduce_dataset=None, return_indices=False, **kwargs_loader):
        """
        
        """
        if not classes:
            classes = set(y)

        self.name = name
        self.return_indices = return_indices

        # filter
        if isinstance(classes, int):
            classes = range(classes)
        self.classes = classes

        self.n_features = X.shape[1]
        self.n_classes = len(classes)

        # 1. extract right classes
        ix = [i for i, yy in enumerate(y) if int(yy) in classes]
        self.data = X[ix]
        self.targets = y[ix]
        # 2. reduce the remaining dataset
        if reduce_dataset:
            self.data = self.data[::reduce_dataset]
            self.targets = self.targets[::reduce_dataset]
        
        self.data = torch.Tensor(self.data).to(torch.device("cuda"))
        self.targets = torch.LongTensor(self.targets).to(torch.device("cuda"))

        self.loader = tud.DataLoader(self, **kwargs_loader)
        return


    def __len__(self):
        return len(self.targets)


    def __getitem__(self, i):
        if self.return_indices:
            return self.data[i], self.targets[i], i
        else:
            return self.data[i], self.targets[i]





class Dataset():

    def __init__(self, name,
            path=PATH_DATA, batch_size=1024, shuffle=False, train=True,
            n_classes=10, reduce_dataset=1, flatten=False, verbose=1,
            return_indices=False):
        """ Constructor.

        """
        self.name = name
        self.return_indices = return_indices

        samples_attr = "data"
        labels_attr = "targets"

        transform =\
            torchvision.transforms.ToTensor() if not flatten else\
            torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: x.reshape(-1))
            ])

        # load the dataset
        super().__init__(
            root=path,
            train=train,
            download=True,
            transform=transform)

        self.n_classes = n_classes

        # reduce dataset
        self.__setattr__(
            samples_attr,
            self.__getattribute__(samples_attr)[::reduce_dataset]
        )
        self.__setattr__(
            labels_attr,
            self.__getattribute__(labels_attr)[::reduce_dataset]
        )

        self.loader = tud.DataLoader(
            dataset=self, batch_size=batch_size, shuffle=shuffle)

        if verbose > 0:
            print(f"{self.name} {'train' if train else 'test'} dataset loaded: {len(self.loader)} batches.")

        return


    def __getitem__(self, index):
        """ Basic __getitem__ method returning data indeces as well.

        """
        data, target = super().__getitem__(index)
        if self.return_indices:
            return data, target, index
        else:
            return data, target




class MNIST(Dataset, torchvision.datasets.MNIST):
    def __init__(self, **kwargs):
        super().__init__("MNIST", **kwargs)




class FashionMNIST(Dataset, torchvision.datasets.FashionMNIST):
    def __init__(self, **kwargs):
        super().__init__("FashionMNIST", **kwargs)
