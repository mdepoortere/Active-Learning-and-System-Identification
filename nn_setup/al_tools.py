import torch
import numpy as np
import nn_setup
from nn_setup.models import create_model
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

    model = create_model(nn_setup.datamaker.create_dataloaders_al(**dataset_config)['train'], **model_config)
    model_param_path = (TrainedModel().ModelStorage & "dataset_config_hash = '{}'".format(
        dataset_hash) & "config_hash = '{}'".format(model_hash)).fetch("model_state")[0]
    model.load_state_dict(torch.load(model_param_path))
    return model
