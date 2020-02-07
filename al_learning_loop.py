import numpy as np
import nnsetup
from nnsetup.datasets import LabeledImageSet
from mlutils.data.datasets import StaticImageSet
from nnsetup.transforms import Normalized
from mlutils.data.transforms import Subsample, ToTensor
from torch.utils.data import Subset, DataLoader
import pickle
import os
import datajoint as dj
dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USERNAME']
dj.config['database.password'] = os.environ['DJ_PASSWORD']
dj.config['enable_python_native_blobs'] = True
schema = dj.schema('mdep_nnfabrik_al_norm_mc', locals())
dj.config['schema_name'] = "mdep_nnfabrik_al_norm_mc"

from nnfabrik.main import *
from nnsetup.estimator import mc_estimate
from nnsetup.models import create_model
from nnsetup.al_tools import load_latest_model, calc_mean_sd




def load_obj(file):
    with open('./data/' + file + '.pkl', 'rb') as f:
        return pickle.load(f)


aa = dict(architect_name="matthias Depoortere",
          email="depoortere.matthias@gmail;com", affiliation="sinzlab", dj_username="mdep")
Fabrikant().insert1(aa, skip_duplicates=True)
dat = StaticImageSet('/notebooks/data/static20892-3-14-preproc0.h5', 'images', 'responses')
Seed().insert([{'seed': 13}], skip_duplicates=True)

labeled_dat = LabeledImageSet('./data/static20892-3-14-preproc0.h5', 'images', 'responses')
idx = (labeled_dat.neurons.area == 'V1') & (labeled_dat.neurons.layer == 'L2/3')
labeled_dat.transforms = [Subsample(np.where(idx)[0]), ToTensor(cuda=True),
                          Normalized(np.where(dat.tiers == 'train')[0], dat.responses, cuda=True)]

TOTAL_IM = np.where(dat.tiers == 'train')[0].size
MAX_IM = TOTAL_IM
N_SAMPLES = 100
N_AQUIRE = 10

selected_idx = np.load('al_idx_1200.npy') #set(np.random.choice(np.where(dat.tiers == 'train')[0], size=n_im, replace=False))
n_im = 1200
all_idx = set(np.where(dat.tiers == 'train')[0])

model_config = load_obj('best_model_config')
model_config['random_seed'] = 5
model_config['gpu_id'] = 0

model_entry = dict(configurator="nnsetup.models.create_model", config_object=model_config,
                   model_architect="Matthias Depoortere", model_comment="Best model on full dataset")
#Model().add_entry(**model_entry)

trainer_config = load_obj('best_train_config')
trainer_entry = dict(training_function="nnsetup.trainer.train_model", training_config=trainer_config,
                     trainer_architect="Matthias Depoortere", trainer_comment="best trainer on full dataset")
#Trainer().add_entry(**trainer_entry)
al_statistics = []

while n_im < MAX_IM:

    dataset_config = dict(file='/notebooks/data/static20892-3-14-preproc0.h5', selected_idx=list(selected_idx),
                          batch_size=64)
    dataset_hash = make_hash(dataset_config)

    dataset_entry = dict(dataset_loader="nnsetup.datamaker.create_dataloaders_al", dataset_config=dataset_config,
                         dataset_architect="Matthias Depoortere", dataset_comment=" Actively grown dataset")
    Dataset().add_entry(**dataset_entry)
    restriction = (
        'dataset_loader in ("{}")'.format("nnsetup.datamaker.create_dataloaders_al"),
        'dataset_config_hash in ("{}")'.format(dataset_hash))
    TrainedModel().populate(*restriction)

    model = load_latest_model(dataset_config, model_config, dataset_hash, model_hash)

    scoring_set = Subset(labeled_dat, list(all_idx - set(selected_idx)))
    scoring_loader = DataLoader(scoring_set, shuffle=True, batch_size=128)
    sample_mean, sample_sd = calc_mean_sd(model, scoring_loader, N_SAMPLES)
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