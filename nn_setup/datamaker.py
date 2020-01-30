from mlutils.data.datasets import StaticImageSet
from mlutils.data.transforms import Subsample, ToTensor
from nn_setup.transforms import Normalized


import numpy as np

from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler


def create_dataloaders(file, seed, batch_size, norm=False, cuda=False):

    dat = StaticImageSet(file, 'images', 'responses')
    idx = (dat.neurons.area == 'V1') & (dat.neurons.layer =='L2/3')
    dat.transforms = [Subsample(np.where(idx)[0]), ToTensor(cuda=cuda)]
    if norm:
        dat.transforms.append(Normalized(np.where(dat.tiers == 'train')[0], dat.responses, cuda=cuda))
    
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


def create_dataloaders_al(file='', seed=0, selected_idx=set([]), batch_size=64, norm=False, cuda=False):
    np.random.seed(seed)
    dat = StaticImageSet(file, 'images', 'responses')
    idx = (dat.neurons.area == 'V1') & (dat.neurons.layer == 'L2/3')
    dat.transforms = [Subsample(np.where(idx)[0]), ToTensor(cuda=True)]
    if norm:
        dat.transforms.append(Normalized(np.where(dat.tiers == 'train')[0], dat.responses, cuda=cuda))

    selected_set = Subset(dat, selected_idx)
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


def create_dataloaders_rand(file, seed, total_im, n_selected, batch_size, norm=True, cuda=True):
    dat = StaticImageSet(file, 'images', 'responses')
    idx = (dat.neurons.area == 'V1') & (dat.neurons.layer == 'L2/3')
    dat.transforms = [Subsample(np.where(idx)[0]), ToTensor(cuda=cuda)]
    if norm:
        dat.transforms.append(Normalized(np.where(dat.tiers == 'train')[0], dat.responses, cuda=cuda))
    np.random.seed(seed)
    selected_indexes = np.random.choice(np.where(dat.tiers == 'train')[0], size=n_selected, replace=False)
    selected_set = Subset(dat, selected_indexes)
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
