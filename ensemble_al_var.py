import numpy as np
from nn_setup.datasets import LabeledImageSet
from mlutils.data.transforms import ToTensor, Subsample
from mlutils.data.datasets import StaticImageSet
from nn_setup.transforms import Normalized
import pickle
import os
import datajoint as dj
from torch.utils.data import Subset, DataLoader
dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USERNAME']
dj.config['database.password'] = os.environ['DJ_PASSWORD']
dj.config['enable_python_native_blobs'] = True
schema = dj.schema('mdep_nnfabrik_al_ens_var', locals())
dj.config['schema_name'] = "mdep_nnfabrik_al_ens_var"

from nnfabrik.main import *

import nn_setup
from nn_setup import datamaker
from estimator import mc_estimate
from nn_setup.models import create_model


def load_obj(file):
    with open('./data/' + file + '.pkl', 'rb') as f:
        return pickle.load(f)


def calc_var(preds):
    vars = np.stack(preds, axis=0).std(axis=0).mean(axis=0)
    return vars

aa = dict(architect_name="matthias Depoortere",
          email="depoortere.matthias@gmail;com", affiliation="sinzlab", dj_username="mdep")
Fabrikant().insert1(aa, skip_duplicates=True)
dat = StaticImageSet('/notebooks/data/static20892-3-14-preproc0.h5', 'images', 'responses')
Seed().insert([{'seed': 13}], skip_duplicates=True)


my_dat = LabeledImageSet('/notebooks/data/static20892-3-14-preproc0.h5', 'images', 'responses')
idx = (my_dat.neurons.area == 'V1') & (my_dat.neurons.layer == 'L2/3')
my_dat.transforms = [Subsample(np.where(idx)[0]), ToTensor(cuda=True),
                          Normalized(np.where(dat.tiers == 'train')[0], dat.responses, cuda=True)]

TOTAL_IM = np.where(dat.tiers == 'train')[0].size
MAX_IM = TOTAL_IM
N_AQUIRE = 50
n_im = 500

selected_idx = np.load("ens_var_selected_idx_3250.npy")
#set(np.random.choice(np.where(dat.tiers == 'train')[0], size=n_im, replace=False))
n_im = len(list(selected_idx))
all_idx = set(np.where(dat.tiers == 'train')[0])
model_hashes = []
for i in range(8):
    model_config = load_obj('best_model_config')
    model_config['random_seed'] = i

    model_hash = make_hash(model_config)
    model_hashes.append(model_hash)
    """model_entry = dict(configurator="nn_setup.models.create_model", config_object=model_config,
                   model_architect="Matthias Depoortere", model_comment="Best model config")
    Model().add_entry(**model_entry)"""

"""trainer_config = load_obj('best_train_config')

trainer_entry = dict(training_function="nn_setup.trainer.train_model", training_config=trainer_config,
                     trainer_architect="Matthias Depoortere", trainer_comment="best trainer on full dataset")
Trainer().add_entry(**trainer_entry)"""
trained_mods = np.load('models_3250_already_trained.npy', allow_pickle=True)
while n_im < MAX_IM:
    if n_im == 3250:
        predictions = []
        collected_labels = []
        model_errors = []
        dataset_config = dict(file='/notebooks/data/static20892-3-14-preproc0.h5', selected_idx=list(selected_idx), batch_size=64, norm=True, cuda=True)
        dataset_hash = make_hash(dataset_config)
        dataset_entry = dict(dataset_loader="nn_setup.datamaker.create_dataloaders_al", dataset_config=dataset_config,
                         dataset_architect="Matthias Depoortere", dataset_comment=" Actively grown dataset")
        print('already added this one')
        for model_hash in model_hashes:
            print('model already trained {}'.format(model_hash))
            model = create_model(nn_setup.datamaker.create_dataloaders_al(**dataset_config)['train'], **model_config)
            model_param_path = (TrainedModel().ModelStorage & "dataset_config_hash = '{}'".format(dataset_hash) & "config_hash = '{}'".format(model_hash)).fetch("model_state")[0]
            model.load_state_dict(torch.load(model_param_path))
            scoring_set = Subset(my_dat, list(all_idx - set(selected_idx)))
            scoring_loader = DataLoader(scoring_set, shuffle=False, batch_size=64)
            preds_model = []
            preds_labels = []
            loss_model = []
            model.eval()
            for batch, labels in scoring_loader:
                with torch.no_grad():
                    out = model(batch)
                    preds_model.append(out.cpu().numpy())
                    preds_labels.append(labels[:, -1].cpu().numpy().astype('int32'))
            preds_model = np.concatenate(preds_model, axis=0)
            preds_labels = np.concatenate(preds_labels, axis=0)
            predictions.append(preds_model)
            collected_labels.append(preds_labels)
        predictions = np.stack(predictions)
        mean_var_neuron = predictions.std(axis=0).mean(axis=1)
        top_n = np.argsort(mean_var_neuron)[-N_AQUIRE:]
        selected_idx = set(list(selected_idx)).union(set(list(collected_labels[0][top_n])))
        print(top_n)
        print(mean_var_neuron, '\n')
        n_im = len(selected_idx)

    else:
        predictions = []
        collected_labels = []
        model_errors = []
        dataset_config = dict(file='/notebooks/data/static20892-3-14-preproc0.h5', selected_idx=list(selected_idx),
                              batch_size=64, norm=True, cuda=True)
        dataset_hash = make_hash(dataset_config)
        dataset_entry = dict(dataset_loader="nn_setup.datamaker.create_dataloaders_al", dataset_config=dataset_config,
                             dataset_architect="Matthias Depoortere", dataset_comment=" Actively grown dataset")
        Dataset().add_entry(**dataset_entry)

        for model_hash in model_hashes:
            restriction = ('config_hash in ("{}")'.format(model_hash),
                           'dataset_loader in ("{}")'.format("nn_setup.datamaker.create_dataloaders_al"),
                           'dataset_config_hash in ("{}")'.format(dataset_hash))
            TrainedModel().populate(*restriction)
            model = create_model(nn_setup.datamaker.create_dataloaders_al(**dataset_config)['train'], **model_config)
            model_param_path = (TrainedModel().ModelStorage & "dataset_config_hash = '{}'".format(
                dataset_hash) & "config_hash = '{}'".format(model_hash)).fetch("model_state")[0]
            model.load_state_dict(torch.load(model_param_path))
            scoring_set = Subset(my_dat, list(all_idx - set(selected_idx)))
            scoring_loader = DataLoader(scoring_set, shuffle=False, batch_size=64)
            preds_model = []
            preds_labels = []
            loss_model = []
            model.eval()
            for batch, labels in scoring_loader:
                with torch.no_grad():
                    out = model(batch)
                    preds_model.append(out.cpu().numpy())
                    preds_labels.append(labels[:, -1].cpu().numpy().astype('int32'))
            preds_model = np.concatenate(preds_model, axis=0)
            preds_labels = np.concatenate(preds_labels, axis=0)
            predictions.append(preds_model)
            collected_labels.append(preds_labels)
        predictions = np.stack(predictions)
        mean_var_neuron = predictions.std(axis=0).mean(axis=1)
        top_n = np.argsort(mean_var_neuron)[-N_AQUIRE:]
        selected_idx = set(list(selected_idx)).union(set(list(collected_labels[0][top_n])))
        n_im = len(selected_idx)
        print(top_n)
        print(mean_var_neuron, '\n')

