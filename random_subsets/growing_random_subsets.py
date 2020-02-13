import os
import sys
sys.path.append('/notebooks')
import pickle
import numpy as np
import datajoint as dj
import nnsetup
from mlutils.data.datasets import StaticImageSet


dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USERNAME']
dj.config['database.password'] = os.environ['DJ_PASSWORD']
dj.config['enable_python_native_blobs'] = True
schema = dj.schema('mdep_random_subsets_synth', locals())
dj.config['schema_name'] = "mdep_random_subsets_synth"

from nnfabrik.main import *
from nnsetup.tables import TrainedModel


def load_obj(file):
    with open('./data/' + file + '.pkl', 'rb') as f:
        return pickle.load(f)


Fabrikant().insert1(dict(fabrikant_name='Matthias Depoortere',
                         email="depoortere.matthias@gmail.com",
                         affiliation='sinzlab',
                         dj_username="mdep"), skip_duplicates=True)

Seed().insert([{'seed': 13}], skip_duplicates=True)

dat = StaticImageSet('/notebooks/toy_data/toy_dataset.hdf5', 'images', 'responses')
TOTAL_IM = np.where(dat.tiers == 'train')[0].size

model_config = load_obj('best_model_config')
model_config['random_seed'] = 5
model_entry = dict(model_fn="nnsetup.models.create_model", model_config=model_config,
                   model_fabrikant="Matthias Depoortere", model_comment="Best model hp on full dataset")
Model().add_entry(**model_entry)

trainer_config = load_obj('best_train_config')
trainer_entry = dict(trainer_fn="nnsetup.trainer.train_model", trainer_config=trainer_config,
                     trainer_fabrikant="Matthias Depoortere", trainer_comment="best trainer on full dataset")
Trainer().add_entry(**trainer_entry)

all_index_sets = []
np.random.seed(12)

for i in np.linspace(500, TOTAL_IM, 150).astype('int32'):
    dataset_config = dict(file='/notebooks/toy_data/toy_dataset.hdf5', seed=i,
                          total_im=TOTAL_IM, n_selected=i, batch_size=64, norm=True, cuda=True)
    dataset_hash = make_hash(dataset_config)

    dataset_entry = dict(dataset_fn="nnsetup.datamaker.create_dataloaders_rand", dataset_config=dataset_config,
                         dataset_fabrikant="Matthias Depoortere", dataset_comment="Randomly sampled subset")
    Dataset().add_entry(**dataset_entry)

    restriction = (
        'dataset_fn in ("{}")'.format("nnsetup.datamaker.create_dataloaders_rand"),
        'dataset_hash in ("{}")'.format(dataset_hash))
    TrainedModel().populate(*restriction)
