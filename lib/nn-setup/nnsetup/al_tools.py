import numpy as np
from mlutils.measures import corr
from nnsetup.models import create_model
from nnsetup.estimator import mc_estimate
import nnsetup.datamaker as datamaker
from nnsetup.tables import TrainedModel
from nnfabrik.main import *

def calc_preds_labels(model, scoring_loader):
    preds_model = []
    preds_labels =  []
    for batch, labels in scoring_loader:
        with torch.no_grad():
            out = model(batch)
            preds_model.append(out.cpu().numpy())
            preds_labels.append(labels[:, -1].cpu().numpy().astype('int32'))
    preds_model = np.concatenate(preds_model, axis=0)
    preds_labels = np.concatenate(preds_labels, axis=0)
    return preds_model, preds_labels


def load_latest_model(dataset_config, model_config, dataset_hash, model_hash):

    model = create_model(datamaker.create_dataloaders_synth(**dataset_config),
                         **model_config)
    model_param_path = (TrainedModel().ModelStorage & "dataset_config_hash = '{}'".format(
        dataset_hash) & "config_hash = '{}'".format(model_hash)).fetch("model_state")[0]
    model.load_state_dict(torch.load(model_param_path))
    return model


def calc_loss_labels(model, scoring_loader, criterion):
    preds_labels = []
    loss_model = []
    model.eval()
    for batch, labels in scoring_loader:
        with torch.no_grad():
            out = model(batch)
            loss = criterion(out, labels[:, :-1]).mean(dim=1).cpu().numpy()
            preds_labels.append(labels[:, -1].cpu().numpy().astype('int32'))
            loss_model.append(loss)
    preds_labels = np.concatenate(preds_labels, axis=0)
    loss_model = np.concatenate(loss_model)
    return loss_model, preds_labels


def calc_mean_sd(model, scoring_loader, N_SAMPLES):

    sample_sd = []
    sample_mean = []
    for batch, labels in scoring_loader:
        mean, sd = mc_estimate(model, batch.cuda(), N_SAMPLES)

        sample_mean.append([labels[:, -1].cpu(), mean])
        sample_sd.append([labels[:, -1].cpu(), sd])
    return sample_mean, sample_sd


def calc_corr_labels(model, scoring_loader):
    preds_labels = []
    corr_model = []
    model.eval()
    for batch, labels in scoring_loader:
        with torch.no_grad():
            out = model(batch).cpu().numpy()
            correlation = corr(out, labels[:, :-1].cpu().numpy())
            preds_labels.append(labels[:, -1].cpu().numpy().astype('int32'))
            corr_model.append(correlation)
    preds_labels = np.concatenate(preds_labels, axis=0)
    corr_model = np.concatenate(corr_model)
    return corr_model, preds_labels
