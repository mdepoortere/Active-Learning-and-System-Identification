import numpy as np
from nnsetup.datasets import LabeledImageSet
from mlutils.data.transforms import ToTensor, Subsample
from mlutils.data.datasets import StaticImageSet
from nnsetup.transforms import Normalized
import pickle
import os
import datajoint as dj
from torch.utils.data import Subset, DataLoader

dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USERNAME']
dj.config['database.password'] = os.environ['DJ_PASSWORD']
dj.config['enable_python_native_blobs'] = True
schema = dj.schema('mdep_ens_toy_var', locals())
dj.config['schema_name'] = "mdep_ens_toy_var"

from nnfabrik.main import *
from nnsetup.tables import TrainedModel
from nnsetup.al_tools import calc_preds_labels, load_latest_model


def load_obj(file):
    with open('./data/' + file + '.pkl', 'rb') as f:
        return pickle.load(f)


aa = dict(fabrikant_name="matthias Depoortere",
          email="depoortere.matthias@gmail;com", affiliation="sinzlab", dj_username="mdep")
Fabrikant().insert1(aa, skip_duplicates=True)
Seed().insert([{'seed': 13}], skip_duplicates=True)


my_dat = LabeledImageSet('/notebooks/toy_data/toy_dataset.hdf5', 'images', 'responses')
#idx = (my_dat.neurons.area == 'V1') & (my_dat.neurons.layer == 'L2/3')
my_dat.transforms = [ToTensor(cuda=True),
                     Normalized(np.where(my_dat.tiers == 'train')[0], my_dat.responses, cuda=True)]


TOTAL_IM = np.where(my_dat.tiers == 'train')[0].size
MAX_IM = TOTAL_IM
N_AQUIRE = 50
n_im = 500

selected_idx = set(np.random.choice(np.where(my_dat.tiers == 'train')[0], size=n_im, replace=False))

all_idx = set(np.where(my_dat.tiers == 'train')[0])

model_hashes = []
for i in range(8):
    model_config = load_obj('best_model_config')
    model_config['random_seed'] = i

    model_hash = make_hash(model_config)
    model_hashes.append(model_hash)
    model_entry = dict(model_fn="nnsetup.models.create_model", model_config=model_config,
                       model_fabrikant="Matthias Depoortere", model_comment="Best model config")
    Model().add_entry(**model_entry)

trainer_config = load_obj('best_train_config')
trainer_config['patience'] = 5

trainer_entry = dict(trainer_fn="nnsetup.trainer.train_model", trainer_config=trainer_config,
                     trainer_fabrikant="Matthias Depoortere", trainer_comment="best trainer on full dataset")
Trainer().add_entry(**trainer_entry)

while n_im < MAX_IM:
    predictions = []
    collected_labels = []
    dataset_config = dict(file='/notebooks/toy_data/toy_dataset.hdf5',
                          selected_idx=list(selected_idx), seed=5, batch_size=64, norm=True, cuda=True)
    dataset_hash = make_hash(dataset_config)
    dataset_entry = dict(dataset_fn="nnsetup.datamaker.create_dataloaders_synth", dataset_config=dataset_config,
                         dataset_fabrikant="Matthias Depoortere", dataset_comment=" Actively grown dataset")
    Dataset().add_entry(**dataset_entry)
    for model_hash in model_hashes:
        restriction = ('model_hash in ("{}")'.format(model_hash),
                       'dataset_fn in ("{}")'.format("nnsetup.datamaker.create_dataloaders_synth"),
                       'dataset_hash in ("{}")'.format(dataset_hash))
        TrainedModel().populate(*restriction)
        model = load_latest_model(dataset_config, model_config, dataset_hash, model_hash)

        scoring_set = Subset(my_dat, list(all_idx - set(selected_idx)))
        scoring_loader = DataLoader(scoring_set, shuffle=False, batch_size=64)
        model.eval()
        preds_model, preds_labels = calc_preds_labels(model, scoring_loader)
        predictions.append(preds_model)
        collected_labels.append(preds_labels)
    predictions = np.stack(predictions)
    mean_var_neuron = predictions.std(axis=0).mean(axis=1)
    top_n = np.argsort(mean_var_neuron)[-N_AQUIRE:]
    selected_idx = set(list(selected_idx)).union(set(list(collected_labels[0][top_n])))
    print(top_n)
    print(mean_var_neuron, '\n')
    n_im = len(selected_idx)
