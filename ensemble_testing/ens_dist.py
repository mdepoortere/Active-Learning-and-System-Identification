import numpy as np
from torch.nn import PoissonNLLLoss
from mlutils.data.datasets import StaticImageSet
from nn_setup.datasets import LabeledImageSet
from mlutils.data.transforms import ToTensor, Subsample
from nn_setup.transforms import Normalized
from nn_setup.al_tools import load_latest_model, calc_loss_labels
import pickle
import os
import datajoint as dj
from torch.utils.data import Subset, DataLoader
dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USERNAME']
dj.config['database.password'] = os.environ['DJ_PASSWORD']
dj.config['enable_python_native_blobs'] = True
schema = dj.schema('mdep_ens_dist', locals())
dj.config['schema_name'] = "mdep_ens_dist"

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

test_images = my_dat[()].images[np.where(my_dat.tiers == "test")]
model_hashes = []

for i in range(8):
    model_config = load_obj('best_model_config')
    model_config['random_seed'] = i
    model_config['gpu_id'] = 0
    model_config['dropout_p'] = 0.7

    model_hash = make_hash(model_config)
    model_hashes.append(model_hash)
    model_entry = dict(configurator="nn_setup.models.create_model", config_object=model_config,
                       model_architect="Matthias Depoortere", model_comment="Best model config")
    Model().add_entry(**model_entry)

trainer_config = load_obj('best_train_config')

trainer_entry = dict(training_function="nn_setup.trainer.train_model", training_config=trainer_config,
                     trainer_architect="Matthias Depoortere", trainer_comment="best trainer on full dataset")
Trainer().add_entry(**trainer_entry)

dataset_config = dict(file='/notebooks/data/static20892-3-14-preproc0.h5', batch_size=64, norm=True, cuda=True)
dataset_hash = make_hash(dataset_config)
dataset_entry = dict(dataset_loader="nn_setup.datamaker.create_dataloaders", dataset_config=dataset_config,
                     dataset_architect="Matthias Depoortere", dataset_comment=" Actively grown dataset")
Dataset().add_entry(**dataset_entry)

ens = []
for model_hash in model_hashes:
    restriction = ('config_hash in ("{}")'.format(model_hash),
                   'dataset_loader in ("{}")'.format("nn_setup.datamaker.create_dataloaders_al"),
                   'dataset_config_hash in ("{}")'.format(dataset_hash))
    TrainedModel().populate(*restriction)
    model = load_latest_model(dataset_config, model_config, dataset_hash, model_hash)
    ens.append(model)

ens_dist = []
for i in range(100):
    preds = []
    for net in ens:
        with torch.no_grad():
            preds.append(net(test_images[i*10].view(1, 1, 36, 64)))
    preds = torch.cat(preds, dim=0)
    ens_dist.append((preds.mean(dim=0), preds.std(dim=0)))

np.save("ens_dist.npy", ens_dist)