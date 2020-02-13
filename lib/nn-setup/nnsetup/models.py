from mlutils.layers.readouts import PointPooled2d
from nnsetup.cores import Stacked2dCoreDropOut
from functools import partial
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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
        outs = [model(x) for i, model in enumerate(self.models)]
        return outs

    @staticmethod
    def diff_real(preds, true):
        diff = 0
        for pred in preds:
            diff += F.mse_loss(pred, true, reduction='mean')
        return diff

    @staticmethod
    def variance(preds):
        return torch.cat(preds, dim=0).std(dim=0).mean()


def create_model(loaders, seed=0, gpu_id=0, **config):
    np.random.seed(seed)
    train_loader = loaders['train']
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
    model = model.to('cuda:{}'.format(gpu_id))
    model.train()
    return model


def create_ensemble(train_loader, **config):

    ensemble = Ensemble(train_loader, config)

    return ensemble