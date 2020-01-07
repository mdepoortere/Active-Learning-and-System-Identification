import numpy as np
from mlutils.data.datasets import StaticImageSet
import pickle
import os
import datajoint as dj
dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USERNAME']
dj.config['database.password'] = os.environ['DJ_PASSWORD']
dj.config['enable_python_native_blobs'] = True
schema = dj.schema('mdep_nnfabrik_al_ensemble', locals())
dj.config['schema_name'] = "mdep_nnfabrik_al_ensemble"

from nnfabrik.main import *

from nn_setup import datamaker
from estimator import mc_estimate
from nn_setup.models import create_ensemble




def load_obj(file):
    with open('./data/' + file + '.pkl', 'rb') as f:
        return pickle.load(f)


aa = dict(architect_name="matthias Depoortere",
          email="depoortere.matthias@gmail;com", affiliation="sinzlab", dj_username="mdep")
Fabrikant().insert1(aa, skip_duplicates=True)
dat = StaticImageSet('data/static20892-3-14-preproc0.h5', 'images', 'responses')
Seed().insert([{'seed': 13}], skip_duplicates=True)


TOTAL_IM = np.where(dat.tiers == 'train')[0].size
MAX_IM = 1000
N_SAMPLES = 16
N_AQUIRE = 50
n_im = 64

selected_idx = set(np.random.randint(0, np.where(dat.tiers == 'train')[0].size, 64))
all_idx = set(range(TOTAL_IM))

for i in range(4):
    model_config = load_obj('best_model_config')
    model_config['seeds'] =
    model_entry = dict(configurator="models.create_ensemble", config_object=model_config,
                   model_architect="Matthias Depoortere", model_comment="Ensemble of models")
Model().add_entry(**model_entry)

trainer_config = load_obj('best_train_config')
trainer_entry = dict(training_function="trainer.train", training_config=trainer_config,
                     trainer_architect="Matthias Depoortere", trainer_comment="best trainer on full dataset")
Trainer().add_entry(**trainer_entry)
al_statistics = []

while n_im < MAX_IM:

    dataset_config = dict(file='/notebooks/data/static20892-3-14-preproc0.h5', selected_idx=list(selected_idx), batch_size=64)
    dataset_hash = make_hash(dataset_config)
    dataset_entry = dict(dataset_loader="datamaker.create_dataloaders_al", dataset_config=dataset_config,
                         dataset_architect="Matthias Depoortere", dataset_comment=" Actively grown dataset")
    Dataset().add_entry(**dataset_entry)
    restriction = (
        'dataset_loader in ("{}")'.format("datamaker.create_dataloaders_al"),
        'dataset_config_hash in ("{}")'.format(dataset_hash))
    TrainedModel().populate(*restriction)

    ensemble = create_ensemble(datamaker.create_dataloaders_al(**dataset_config)['train'], Seed.fetch('seed'), **model_config)
    model_param_path = (TrainedModel().ModelStorage & "dataset_config_hash = '{}'".format(dataset_hash)).fetch("model_state")[0]
    ensemble.load_state_dict(torch.load(model_param_path))



np.save('Best_subset', selected_idx)
np.save('statistics', al_statistics)
