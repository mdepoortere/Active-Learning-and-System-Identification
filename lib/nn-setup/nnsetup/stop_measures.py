from contextlib import contextmanager
import numpy as np
import warnings
from mlutils.measures import corr
from scipy import stats
from nnsetup.estimator import mean_estimate


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


def compute_predictions_mc(loader, model):
    y, y_hat = [], []
    with eval_state_mc(model):
        for x_val, y_val in loader:
            mean = mean_estimate(model, x_val, 5)
            y_hat.append(mean.detach().cpu().numpy())
            y.append(y_val.detach().cpu().numpy())
    y, y_hat = map(np.vstack, (y, y_hat))
    return y, y_hat


def corr_stop_mc(model, val_loader):
    with eval_state_mc(model):
        y, y_hat = compute_predictions_mc(val_loader, model)

    ret = corr(y, y_hat, axis=0)

    if np.any(np.isnan(ret)):
        warnings.warn('{}% NaNs '.format(np.isnan(ret).mean() * 100))
    ret[np.isnan(ret)] = 0

    return ret.mean()


def gamma_stop_mc(model, val_loader):
    with eval_state_mc(model):
        y, y_hat = compute_predictions_mc(val_loader, model)

    ret = -stats.gamma.logpdf(y + 1e-7, y_hat + 0.5).mean(axis=1) / np.log(2)
    if np.any(np.isnan(ret)):
        warnings.warn(' {}% NaNs '.format(np.isnan(ret).mean() * 100))
    ret[np.isnan(ret)] = 0
    # -- average if requested
    return ret.mean()


def exp_stop_mc(model, val_loader, bias=1e-12, target_bias=1e-7):
    with eval_state_mc(model):
        y, y_hat = compute_predictions_mc(val_loader, model)
    y = y + target_bias
    y_hat = y_hat + bias
    ret = (y / y_hat + np.log(y_hat)).mean(axis=1) / np.log(2)
    if np.any(np.isnan(ret)):
        warnings.warn(' {}% NaNs '.format(np.isnan(ret).mean() * 100))
    ret[np.isnan(ret)] = 0
    # -- average if requested
    return ret.mean()


def poisson_stop_mc(model, val_loader):
    with eval_state_mc(model):
        target, output = compute_predictions_mc(val_loader, model)

    ret = (output - target * np.log(output + 1e-12))
    if np.any(np.isnan(ret)):
        warnings.warn(' {}% NaNs '.format(np.isnan(ret).mean() * 100))
    ret[np.isnan(ret)] = 0
    # -- average if requested
    return ret.mean()


def full_objective(inputs, targets, model, criterion):
    return criterion(model(inputs), targets) + model.core.regularizer()
