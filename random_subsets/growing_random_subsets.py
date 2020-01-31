import os
import sys
sys.path.append('/notebooks')
import pickle
import numpy as np
import datajoint as dj
import nn_setup
from mlutils.data.datasets import StaticImageSet


dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USERNAME']
dj.config['database.password'] = os.environ['DJ_PASSWORD']
dj.config['enable_python_native_blobs'] = True
schema = dj.schema('mdep_random_subsets_growing_norm', locals())
dj.config['schema_name'] = "mdep_random_subsets_growing_norm"

from nnfabrik.main import *


def load_obj(file):
    with open('./data/' + file + '.pkl', 'rb') as f:
        return pickle.load(f)


Fabrikant().insert1(dict(architect_name='Matthias Depoortere',
                         email="depoortere.matthias@gmail.com",
                         affiliation='sinzlab',
                         dj_username="mdep"), skip_duplicates=True)

Seed().insert([{'seed': 13}], skip_duplicates=True)

dat = StaticImageSet('/notebooks/data/static20892-3-14-preproc0.h5', 'images', 'responses')
TOTAL_IM = np.where(dat.tiers == 'train')[0].size

model_config = load_obj('best_model_config')
model_config['random_seed'] = 5
model_entry = dict(configurator="nn_setup.models.create_model", config_object=model_config,
                   model_architect="Matthias Depoortere", model_comment="Best model hp on full dataset")
#Model().add_entry(**model_entry)

trainer_config = load_obj('best_train_config')
trainer_entry = dict(training_function="nn_setup.trainer.train_model", training_config=trainer_config,
                     trainer_architect="Matthias Depoortere", trainer_comment="best trainer on full dataset")
#Trainer().add_entry(**trainer_entry)

all_index_sets = []
hashes = []
np.random.seed(12)

for i in np.linspace(1434, TOTAL_IM, 150).astype('int32'):
    dataset_config = dict(file='/notebooks/data/static20892-3-14-preproc0.h5', seed=i,
                          total_im=TOTAL_IM, n_selected=i, batch_size=64, norm=True, cuda=True)
    dataset_hash = make_hash(dataset_config)

    dataset_entry = dict(dataset_loader="nn_setup.datamaker.create_dataloaders_rand", dataset_config=dataset_config,
                         dataset_architect="Matthias Depoortere", dataset_comment="Randomly sampled subset")
    Dataset().add_entry(**dataset_entry)

    restriction = (
        'dataset_loader in ("{}")'.format("nn_setup.datamaker.create_dataloaders_rand"),
        'dataset_config_hash in ("{}")'.format(dataset_hash))
    TrainedModel().populate(*restriction)
