from torch.utils.data import Subset, DataLoader
import numpy as np
from mlutils.data.datasets import StaticImageSet
import pickle
import os
import datajoint as dj
dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USERNAME']
dj.config['database.password'] = os.environ['DJ_PASSWORD']
dj.config['enable_python_native_blobs'] = True
schema = dj.schema('mdep_nnfabrik_mc_al', locals())
dj.config['schema_name'] = "mdep_nnfabrik_mc_al"

from nnfabrik.main import *
import nn_setup
from estimator import mc_estimate
from nn_setup.models import create_model




def load_obj(file):
    with open('./data/' + file + '.pkl', 'rb') as f:
        return pickle.load(f)


aa = dict(architect_name="matthias Depoortere",
          email="depoortere.matthias@gmail;com", affiliation="sinzlab", dj_username="mdep")
Fabrikant().insert1(aa, skip_duplicates=True)
dat = StaticImageSet('/notebooks/data/static20892-3-14-preproc0.h5', 'images', 'responses')
Seed().insert([{'seed': 13}], skip_duplicates=True)


class LabeledData(torch.utils.data.Dataset):
    """ Add the indexes of the datapoints as the last column in the response vector
    """
    def __init__(self, dat,  im, responses):
        super().__init__()
        self.train_idx = np.where(dat.tiers == 'train')[0]
        self.im = im[self.train_idx]
        self.responses = torch.tensor(np.hstack([responses[self.train_idx],
                                                 np.arange(0, np.where(dat.tiers == 'train')[0].size).reshape(-1, 1)]))

    def __getitem__(self, index):
        return self.im[index], self.responses[index]

    def __len__(self):
        return self.images
my_dat = LabeledData(dat, dat.images, dat.responses)

TOTAL_IM = np.where(dat.tiers == 'train')[0].size
MAX_IM = TOTAL_IM
N_SAMPLES = 100
N_AQUIRE = 20
n_im = 400

selected_idx = np.random.choice(np.arange(TOTAL_IM), n_im)
all_idx = set(range(TOTAL_IM))

model_config = load_obj('best_model_config')
model_entry = dict(configurator="nn_setup.models.create_model", config_object=model_config,
                   model_architect="Matthias Depoortere", model_comment="Best model on full dataset")
Model().add_entry(**model_entry)

trainer_config = load_obj('best_train_config')
trainer_entry = dict(training_function="nn_setup.trainer.train_model", training_config=trainer_config,
                     trainer_architect="Matthias Depoortere", trainer_comment="best trainer on full dataset")
Trainer().add_entry(**trainer_entry)
al_statistics = []

while n_im < MAX_IM:

    dataset_config = dict(file='/notebooks/data/static20892-3-14-preproc0.h5', selected_idx=list(selected_idx), batch_size=64)
    dataset_hash = make_hash(dataset_config)

    dataset_entry = dict(dataset_loader="nn_setup.datamaker.create_dataloaders_al", dataset_config=dataset_config,
                     dataset_architect="Matthias Depoortere", dataset_comment=" Actively grown dataset")
    Dataset().add_entry(**dataset_entry)
    restriction = (
        'dataset_loader in ("{}")'.format("nn_setup.datamaker.create_dataloaders_al"),
        'dataset_config_hash in ("{}")'.format(dataset_hash))
    TrainedModel().populate(*restriction)

    model = create_model(nn_setup.datamaker.create_dataloaders_al(**dataset_config)['train'], Seed.fetch('seed'), **model_config)
    model_param_path = (TrainedModel().ModelStorage & "dataset_config_hash = '{}'".format(dataset_hash)).fetch("model_state")[0]
    model.load_state_dict(torch.load(model_param_path))

    sample_sd = []
    sample_mean = []
    scoring_set = Subset(my_dat, list(all_idx - set(selected_idx)))
    scoring_loader = DataLoader(scoring_set, shuffle=True, batch_size=128)
    for batch, labels in scoring_loader:
        mean, sd = mc_estimate(model, batch.cuda(), N_SAMPLES)

        sample_mean.append([labels[:, -1], mean])
        sample_sd.append([labels[:, -1], sd])

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

    al_statistics.append([sample_mean, sample_sd])
np.save('statistics', al_statistics)