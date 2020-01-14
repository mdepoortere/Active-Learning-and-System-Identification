import numpy as np
import torch
from contextlib import contextmanager

@contextmanager
def eval_state_mc(model):
    training_status = model.training
    try:
        model.eval()
        for feature in model.core.features:
            if feature.drop:
                feature.drop.training = True
        yield model
    finally:
        model.train(training_status)



@contextmanager
def eval_state_ensemble(ensemble):
    training_status = ensemble.training
    try:
        ensemble.eval()
        for model in ensemble.models:
            for feature in model.core.features:
                if feature.drop:
                    feature.drop.training = True
        yield ensemble
    finally:
        ensemble.train(training_status)


def mc_estimate(model, x, n_samples):
    with torch.no_grad():
        with eval_state_mc(model):
            samples_batch = torch.stack([model(x) for _ in range(n_samples)], dim=0)
            mean = torch.mean(samples_batch, dim=0).cpu()
            sd = torch.std(samples_batch, dim=[0]).cpu()
            sd = torch.mean(sd, dim=1)

    return mean, sd


def mean_estimate(model, x, n_samples):
    with torch.no_grad():
        with eval_state_mc(model):
            samples_batch = torch.stack([model(x) for _ in range(n_samples)], dim=0)
            mean = torch.mean(samples_batch, dim=0).cpu()

    return mean


def mean_estimate_ens(model, x, n_samples, gpu_id):
    with torch.no_grad():
        with eval_state_mc(model):
            samples_batch = torch.stack([model(x.to('cuda:{}'.format(gpu_id))) for _ in range(n_samples)], dim=0)
            mean = torch.mean(samples_batch, dim=0).cpu()

    return mean

