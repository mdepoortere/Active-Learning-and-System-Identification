import numpy as np
from nn_setup.datasets import LabeledImageSet
from mlutils.data.transforms import ToTensor, Subsample
from mlutils.data.datasets import StaticImageSet
from nn_setup.transforms import Normalized
import pickle
import os
import datajoint as dj
from torch.utils.data import Subset, DataLoader
from nn_setup.al_tools import calc_preds_labels, load_latest_model

dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USERNAME']
dj.config['database.password'] = os.environ['DJ_PASSWORD']
dj.config['enable_python_native_blobs'] = True
schema = dj.schema('mdep_nnfabrik_al_ens_var', locals())
dj.config['schema_name'] = "mdep_nnfabrik_al_ens_var"

from nnfabrik.main import *


def load_obj(file):
    with open('./data/' + file + '.pkl', 'rb') as f:
        return pickle.load(f)


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

selected_idx = set(np.random.choice(np.where(dat.tiers == 'train')[0], size=n_im, replace=False))

all_idx = set(np.where(dat.tiers == 'train')[0])

model_hashes = []
for i in range(8):
    model_config = load_obj('best_model_config')
    model_config['random_seed'] = i

    model_hash = make_hash(model_config)
    model_hashes.append(model_hash)
    model_entry = dict(configurator="nn_setup.models.create_model", config_object=model_config,
                   model_architect="Matthias Depoortere", model_comment="Best model config")
    Model().add_entry(**model_entry)

trainer_config = load_obj('best_train_config')

trainer_entry = dict(training_function="nn_setup.trainer.train_model", training_config=trainer_config,
                     trainer_architect="Matthias Depoortere", trainer_comment="best trainer on full dataset")
Trainer().add_entry(**trainer_entry)

while n_im < MAX_IM:

    predictions = []
    collected_labels = []
    dataset_config = dict(file='/notebooks/data/static20892-3-14-preproc0.h5', selected_idx=list(selected_idx), batch_size=64, norm=True, cuda=True)
    dataset_hash = make_hash(dataset_config)
    dataset_entry = dict(dataset_loader="nn_setup.datamaker.create_dataloaders_al", dataset_config=dataset_config,
                     dataset_architect="Matthias Depoortere", dataset_comment=" Actively grown dataset")
    Dataset().add_entry(**dataset_entry)
    for model_hash in model_hashes:
        restriction = ('config_hash in ("{}")'.format(model_hash),
                       'dataset_loader in ("{}")'.format("nn_setup.datamaker.create_dataloaders_al"),
                       'dataset_config_hash in ("{}")'.format(dataset_hash))
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
