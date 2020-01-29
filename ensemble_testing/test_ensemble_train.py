#!/usr/bin/env python

import sys
import logging
sys.path.append('/notebooks')
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlutils.measures import corr, PoissonLoss
import warnings
from mlutils.layers.readouts import PointPooled2d
from mlutils.layers.cores import Stacked2dCoreDropOut
from functools import partial
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
from nn_setup.stop_measures import eval_state_mc

from nn_setup.stop_measures import full_objective
from nn_setup.trainer import train_ensemble
from nn_setup.datamaker import create_dataloaders
import multiprocessing as mp

from mlutils.training import early_stopping, MultipleObjectiveTracker
from functools import partial


def regularizer(readout, gamma_readout):
    return readout.feature_l1() * gamma_readout

class Encoder(nn.Module):

    def __init__(self, core, readout):
        super().__init__()
        self.core = core
        self.readout = readout

    @staticmethod
    def get_readout_in_shape(core, shape):
        train_state = core.training
        core.eval()
        tmp = torch.Tensor(*shape).normal_()
        nout = core(tmp).size()[1:]
        core.train(train_state)
        return nout

    def forward(self, x):
        x = self.core(x)
        x = self.readout(x)
        return F.elu(x) + 1


class Ensemble(nn.Module):
    def __init__(self,train_loader, config):
        super().__init__()
        self.seeds = config['seeds']
        del config['seeds']
        self.config = config
        self.train_loader = train_loader
        self.n_models = config['n_models']
        self.models = nn.ModuleList()
        for i in range(self.n_models):
            self.models.append(create_model(self.train_loader,
                                            self.seeds[i],  **self.config))
        del self.train_loader
    def forward(self, x):
        outs = [model(x.to('cuda:{}'.format(i))) for i, model in enumerate(self.models)]
        return outs

    @staticmethod
    def diff_real(preds, true):
        return F.mse_loss(preds, true, reduction='mean')

    @staticmethod
    def variance(preds):
        return preds.std(dim=0)


def create_model(train_loader, seed=0, **config):
    np.random.seed(seed)
    in_shape = train_loader.img_shape
    n_neurons = train_loader.n_neurons
    transformed_mean = train_loader.transformed_mean
    core = Stacked2dCoreDropOut(input_channels=1,
                                hidden_channels=32,
                                input_kern=15,
                                hidden_kern=7,
                                dropout_p=config['dropout_p'],
                                layers=3,
                                gamma_hidden=config['gamma_hidden'],
                                gamma_input=config['gamma_input'],
                                skip=3,
                                final_nonlinearity=False,
                                bias=False,
                                momentum=0.9,
                                pad_input=False,
                                batch_norm=True,
                                hidden_dilation=1,
                                laplace_padding=0,
                                input_regularizer="LaplaceL2norm")
    ro_in_shape = Encoder.get_readout_in_shape(core, in_shape)

    readout = PointPooled2d(ro_in_shape, n_neurons,
                            pool_steps=2, pool_kern=4,
                            bias=True, init_range=0.2)

    gamma_readout = 0.1


    readout.regularizer = partial(regularizer, readout, gamma_readout)


    ## Model init
    model = Encoder(core, readout)
    r_mean = transformed_mean
    model.readout.bias.data = r_mean
    model.core.initialize()
    model.train()
    return model


def create_ensemble(train_loader, **config):

    ensemble = Ensemble(train_loader, config)

    return ensemble


from mlutils.data.datasets import StaticImageSet
from mlutils.data.transforms import Subsample, ToTensor

import numpy as np

from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler


"""def create_dataloaders(file, batch_size):
    dat = StaticImageSet(file, 'images', 'responses')
    idx = (dat.neurons.area == 'V1') & (dat.neurons.layer == 'L2/3')
    dat.transforms = [Subsample(np.where(idx)[0]), ToTensor(cuda=False)]

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

model_config = dict(dropout_p=0.5, gamma_hidden=100, gamma_input=1 )
model_config['seeds'] =[5, 6]
model_config['n_models'] =2


def mean_estimate(model, x, n_samples, gpu_id):
    with torch.no_grad():
        with eval_state_mc(model):
            samples_batch = torch.stack([model(x.to('cuda:{}'.format(gpu_id))) for _ in range(n_samples)], dim=0)
            mean = torch.mean(samples_batch, dim=0).cpu()

    return mean


def compute_predictions_mc(loader, model, gpu_id):
    y, y_hat = [], []
    with eval_state_mc(model):
        for x_val, y_val in loader:
            mean = mean_estimate(model, x_val, 5, gpu_id)
            y_hat.append(mean.detach().cpu().numpy())
            y.append(y_val.detach().cpu().numpy())
    y, y_hat = map(np.vstack, (y, y_hat))
    return y, y_hat

def corr_stop_mc(model, val_loader, gpu_id):
    with eval_state_mc(model):
        y, y_hat = compute_predictions_mc(val_loader, model, gpu_id)

    ret = corr(y, y_hat, axis=0)

    if np.any(np.isnan(ret)):
        warnings.warn('{}% NaNs '.format(np.isnan(ret).mean() * 100))
    ret[np.isnan(ret)] = 0

    return ret.mean()

def poisson_stop_mc(model, val_loader, gpu_id):
    with eval_state_mc(model):
        target, output = compute_predictions_mc(val_loader, model, gpu_id)

    ret = (output - target * np.log(output + 1e-12))
    if np.any(np.isnan(ret)):
        warnings.warn(' {}% NaNs '.format(np.isnan(ret).mean() * 100))
    ret[np.isnan(ret)] = 0
    # -- average if requested
    return ret.mean()

def run_model_ens(model, gpu_id, criterion, optimizer, scheduler, stop_closure, train_loader,
        epoch, interval, patience, max_iter, maximize, tolerance,
        restore_best, tracker):
    for epoch, val_obj in early_stopping(model, stop_closure,
                                         interval=interval, patience=patience,
                                         start=epoch, max_iter=max_iter, maximize=maximize,
                                         tolerance=tolerance, restore_best=restore_best,
                                         tracker=tracker):
        scheduler.step(val_obj)
        for images, responses in train_loader:
            optimizer.zero_grad()
            loss = full_objective(images.float().to('cuda:{}'.format(gpu_id)), responses.float().to('cuda:{}'.format(gpu_id)), model, criterion)
            loss.backward()
            optimizer.step()
        print('Epoch {}, Training loss: {} for model{}'.format(epoch, loss, gpu_id))
        optimizer.zero_grad()

    return model, epoch

def train_model_ens(model, seed, **config):
    print('started')
    model.to('cuda:{}'.format(config['gpu_id']))
    loaders = create_dataloaders('/notebooks/data/static20892-3-14-preproc0.h5', batch_size=64)
    train, val, test = loaders['train'],loaders['val'], loaders['test']
    tracker = MultipleObjectiveTracker(
        poisson=partial(poisson_stop_mc, model, val, config['gpu_id']),
        correlation=partial(corr_stop_mc, model, val, config['gpu_id']),
                        )
    print('loaders, tracker init for model {}'.format(config['gpu_id']))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    print('optimizer {}'.format(config['gpu_id']))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=0.2,
                                                           patience=5,
                                                           threshold=1e-3,
                                                           min_lr=1e-4,
                                                           )
    print('scheduler {}'.format(config['gpu_id']))

    stop_closure = lambda model: corr_stop_mc(model, val, config['gpu_id'])
    model, epoch = run_model_ens(model=model,
                                 gpu_id=config['gpu_id'],
                                 criterion=PoissonLoss(avg=False),
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 stop_closure=stop_closure,
                                 train_loader=train,
                                 epoch=0,
                                 interval=1,
                                 patience=10,
                                 max_iter=config['max_iter'],
                                 maximize=True,
                                 tolerance=1e-5,
                                 restore_best=True,
                                 tracker=tracker,
                                 )
    tracker.finalize()
    config['manager_list'].append([np.max(tracker.log['correlation']), tracker.log, config['gpu_id']])


def split_training(ens, model_ids, seed, **config):
    print('start split training')
    manager = mp.Manager()
    results = manager.list()
    config['manager_list'] = results
    processes = []
    for gpu_id, model in enumerate(ens.models[model_ids:model_ids + 2]):
        config['gpu_id'] = gpu_id
        p = mp.Process(target=train_model_ens, args=(model, 5,),
                       kwargs=config)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    all_results = [result for result in results]
    return all_results


def train_ensemble(ensemble, seed, **config):
    all_results = []
    for i in range(0, 10, 2):
        res = split_training(ensemble, i, seed, **config)
        all_results.append(res)
    return all_results"""


if __name__ == '__main__':
    mp.log_to_stderr(logging.DEBUG)
    model_config = dict(dropout_p=0.5, gamma_hidden=100, gamma_input=1)
    model_config['seeds'] = list(range(10))
    model_config['n_models'] = 10

    loaders = create_dataloaders('/notebooks/data/static20892-3-14-preproc0.h5', 5, batch_size=64, norm=True)

    ens = create_ensemble(loaders['train'], **model_config)
    train_config = dict(lr=0.01, weight_decay=0.0000001, max_iter=1, n_gpu=2)
    mp.set_start_method('spawn')
    all_results = train_ensemble(ens, 5, **train_config)
    print('Training has ended \n \n')
    print(all_results[0], '\n')
    #print([(x[0][0], x[1][0]) for x in all_results[1]])
