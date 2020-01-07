import os
import numpy as np
import datajoint as dj


dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USERNAME']
dj.config['database.password'] = os.environ['DJ_PASSWORD']
dj.config['enable_python_native_blobs'] = True
dj.config['schema_name'] = "mdep_autobayes"

from nnfabrik.main import *
from hp.bayesian_search import Bayesian


Fabrikant().insert1(dict(architect_name='Matthias Depoortere',
                         email="depoortere.matthias@gmail.com",
                         affiliation='sinzlab',
                         dj_username="mdep"), skip_duplicates=True)
Seed().insert([{'seed':7}], skip_duplicates=True)

dataset_fn = "datamaker.create_dataloaders"
dataset_config = dict(file='data/static20892-3-14-preproc0.h5')
dataset_config_auto = dict(batch_size={"type": "choice", "values": list(range(8, 128)), "log_scale": False})

model_fn = "model.create_model"
model_config = dict()
model_config_auto = dict(gamma_hidden={"type": "range", "bounds": [1e-3, 1.], "log_scale": True},
                         gamma_input={"type": "range", "bounds": [1e-1, 1e3], "log_scale": True},
                         dropout_p={"type": "range", "bounds": [1e-3, 1.], "log_scale": False})

trainer_fn = "trainer.train"
trainer_config = dict()
trainer_config_auto = dict(
              lr={"type": "range", "bounds": [1e-6, 1e-1], "log_scale": True},
              weight_decay={"type": "range", "bounds": [1e-8, 1e-2], "log_scale": True},
              max_iter={"type": "choice", "values": list(range(25, 128)), "log_scale": False}
              )

aa = Bayesian(dataset_fn, dataset_config, dataset_config_auto,
              model_fn, model_config, model_config_auto,
              trainer_fn, trainer_config, trainer_config_auto, "Matthias Depoortere", total_trials=200)

best_parameters, _, _, _ = aa.run()
np.save("best_params_attempt4c", best_parameters)