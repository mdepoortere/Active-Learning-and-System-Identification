import numpy as np
from mlutils.data.datasets import StaticImageSet, LabeledImageSet
import pickle
import os
import datajoint as dj
from torch.utils.data import Subset, DataLoader
dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USERNAME']
dj.config['database.password'] = os.environ['DJ_PASSWORD']
dj.config['enable_python_native_blobs'] = True
schema = dj.schema('mdep_nnfabrik_al_ensemble', locals())
dj.config['schema_name'] = "mdep_nnfabrik_al_ensemble"

from nnfabrik.main import *

import nn_setup
from nn_setup import datamaker
from estimator import mc_estimate
from nn_setup.models import create_ensemble


def load_obj(file):
    with open('./data/' + file + '.pkl', 'rb') as f:
        return pickle.load(f)


aa = dict(architect_name="matthias Depoortere",
          email="depoortere.matthias@gmail;com", affiliation="sinzlab", dj_username="mdep")
Fabrikant().insert1(aa, skip_duplicates=True)
dat = StaticImageSet('/notebooks/data/static20892-3-14-preproc0.h5', 'images', 'responses')
Seed().insert([{'seed': 13}], skip_duplicates=True)


my_dat = LabeledImageSet('/notebooks/data/static20892-3-14-preproc0.h5', 'images', 'responses')


TOTAL_IM = np.where(dat.tiers == 'train')[0].size
MAX_IM = 1000
N_SAMPLES = 16
N_AQUIRE = 50
n_im = 64

selected_idx = set(np.random.randint(0, np.where(dat.tiers == 'train')[0].size, 64))
all_idx = set(range(TOTAL_IM))


model_config = load_obj('best_model_config')
model_config['seeds'] = range(10)
model_entry = dict(configurator="models.create_ensemble", config_object=model_config,
               model_architect="Matthias Depoortere", model_comment="Ensemble of models")
Model().add_entry(**model_entry)

trainer_config = load_obj('best_train_config')
trainer_entry = dict(training_function="trainer.train_ensemble", training_config=trainer_config,
                     trainer_architect="Matthias Depoortere", trainer_comment="best trainer on full dataset")
Trainer().add_entry(**trainer_entry)
al_statistics = []

while n_im < MAX_IM:

    dataset_config = dict(file='/notebooks/data/static20892-3-14-preproc0.h5', selected_idx=list(selected_idx), batch_size=64)
    dataset_hash = make_hash(dataset_config)
    dataset_entry = dict(dataset_loader="datamaker.create_dataloaders", dataset_config=dataset_config,
                         dataset_architect="Matthias Depoortere", dataset_comment=" Actively grown dataset")
    Dataset().add_entry(**dataset_entry)
    restriction = (
        'dataset_loader in ("{}")'.format("datamaker.create_dataloaders"),
        'dataset_config_hash in ("{}")'.format(dataset_hash))
    TrainedModel().populate(*restriction)
    ens = create_ensemble(nn_setup.datamaker.create_dataloaders_al(**dataset_config)['train'], Seed.fetch('seed'), **model_config)
    ens_param_path = (TrainedModel().ModelStorage & "dataset_config_hash = '{}'".format(dataset_hash)).fetch("model_state")[0]
    ens.load_state_dict(torch.load(ens_param_path))
    sample_var = []
    sample_error = []
    scoring_set = Subset(my_dat, list(all_idx - set(selected_idx)))
    scoring_loader = DataLoader(scoring_set, shuffle=True, batch_size=128)
    for batch, labels in scoring_loader:
        out = ens(batch)
        var = ens.variance(out)
        error = ens.diff_real(out, labels)
        sample_var.append([labels[:, -1], var])
        sample_error.append([labels[:, -1], sd])

    indexes = []
    sd = []
    for element in sample_sd:
        indexes.append(element[0].view(-1, 1))
        sd.append(element[1].view(-1, 1))
    indexes = torch.cat(indexes)
    sd = torch.cat(sd)
    top_n_sd = sd.topk(N_AQUIRE, dim=0)
    top_n_idx = indexes[top_n_sd[1]].flatten().numpy().astype('int32')
    selected_idx = set(list(selected_idx)).union(set(list(top_n_idx)))
    n_im = len(selected_idx)

np.save('Best_subset', selected_idx)
np.save('statistics', al_statistics)
