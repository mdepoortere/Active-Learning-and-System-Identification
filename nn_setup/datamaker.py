from mlutils.data.datasets import StaticImageSet
from mlutils.data.transforms import Subsample, ToTensor, Normalized


import numpy as np
import torch

from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


def create_dataloaders(file, seed, batch_size, cuda=False):

    dat = StaticImageSet(file, 'images', 'responses')
    idx = (dat.neurons.area == 'V1') & (dat.neurons.layer =='L2/3')
    dat.transforms = [Subsample(np.where(idx)[0]), ToTensor(cuda=cuda),
                      Normalized(np.where(dat.tiers == 'train')[0], dat.responses, cuda=cuda)]
    
    train_loader = DataLoader(dat,
                              sampler=SubsetRandomSampler(np.where(dat.tiers == 'train')[0]),
                              batch_size=batch_size)
    train_loader.img_shape = dat.img_shape
    train_loader.n_neurons = dat.n_neurons
    _, train_loader.transformed_mean = dat.transformed_mean()
    
    val_loader = DataLoader(dat,
                              sampler=SubsetRandomSampler(np.where(dat.tiers == 'validation')[0]),
                              batch_size=batch_size)
    val_loader.img_shape = dat.img_shape
    val_loader.n_neurons = dat.n_neurons

    test_loader = DataLoader(dat,
                              sampler=SubsetRandomSampler(np.where(dat.tiers == 'test')[0]),
                              batch_size=batch_size)
    
    test_loader.img_shape = dat.img_shape
    test_loader.n_neurons = dat.n_neurons
    
    loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    return loaders


def create_dataloaders_al(file='', seed=0, selected_idx=set([]), batch_size=64):
    np.random.seed(seed)
    dat = StaticImageSet(file, 'images', 'responses')
    idx = (dat.neurons.area == 'V1') & (dat.neurons.layer == 'L2/3')
    dat.transforms = [Subsample(np.where(idx)[0]), ToTensor(cuda=True),
                      Normalized(np.where(dat.tiers == 'train')[0], dat.responses, cuda=True)]

    train_set = Subset(dat, np.where(dat.tiers == 'train')[0])
    selected_set = Subset(train_set, selected_idx)
    train_loader = DataLoader(selected_set,
                              batch_size=batch_size)
    train_loader.img_shape = dat.img_shape
    train_loader.n_neurons = dat.n_neurons
    _, train_loader.transformed_mean = dat.transformed_mean()

    val_loader = DataLoader(dat,
                            sampler=SubsetRandomSampler(np.where(dat.tiers == 'validation')[0]),
                            batch_size=batch_size)
    val_loader.img_shape = dat.img_shape
    val_loader.n_neurons = dat.n_neurons

    test_loader = DataLoader(dat,
                             sampler=SubsetRandomSampler(np.where(dat.tiers == 'test')[0]),
                             batch_size=batch_size)

    test_loader.img_shape = dat.img_shape
    test_loader.n_neurons = dat.n_neurons

    loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    return loaders


def create_dataloaders_rand(file, seed, total_im, n_selected, batch_size):
    dat = StaticImageSet(file, 'images', 'responses')
    idx = (dat.neurons.area == 'V1') & (dat.neurons.layer == 'L2/3')
    dat.transforms = [Subsample(np.where(idx)[0]), ToTensor(cuda=True),
                      Normalized(np.where(dat.tiers == 'train')[0], dat.responses, cuda=True)]
    np.random.seed(seed)
    selected_indexes = np.random.choice(np.arange(total_im), n_selected + 477, replace=False)
    train_set = Subset(dat, np.where(dat.tiers == 'train')[0])
    selected_set = Subset(train_set, selected_indexes)
    train_loader = DataLoader(selected_set,
                              batch_size=batch_size)

    train_loader.img_shape = dat.img_shape
    train_loader.n_neurons = dat.n_neurons
    _, train_loader.transformed_mean = dat.transformed_mean()

    val_loader = DataLoader(dat,
                            sampler=SubsetRandomSampler(np.where(dat.tiers == 'validation')[0]),
                            batch_size=batch_size)
    val_loader.img_shape = dat.img_shape
    val_loader.n_neurons = dat.n_neurons

    test_loader = DataLoader(dat,
                             sampler=SubsetRandomSampler(np.where(dat.tiers == 'test')[0]),
                             batch_size=batch_size)

    test_loader.img_shape = dat.img_shape
    test_loader.n_neurons = dat.n_neurons

    loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    return loaders
