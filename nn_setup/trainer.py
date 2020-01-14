from functools import partial
import numpy as np
import multiprocessing as mp
import torch
from nn_setup.stop_measures import corr_stop_mc, poisson_stop_mc, gamma_stop_mc, exp_stop_mc, full_objective
from nn_setup.stop_measures import corr_stop_mc_ens, poisson_stop_mc_ens
from nn_setup.datamaker import create_dataloaders
from mlutils.training import early_stopping, MultipleObjectiveTracker
from mlutils.measures import PoissonLoss


def group_models(lijst, n_gpu):
    max_size = n_gpu
    len_list = len(lijst)
    grouping = [list(lijst[i*max_size : (i+1)*max_size]) for i in range(len_list//max_size)]
    if len_list % max_size !=0:
        grouping.append(list(lijst[len(lijst)//max_size * max_size : ]))
    return grouping


def run(model, criterion, optimizer, scheduler, stop_closure, train_loader,
        epoch, interval, patience, max_iter, maximize, tolerance,
        restore_best, tracker):
    for epoch, val_obj in early_stopping(model, stop_closure,
                                         interval=interval, patience=patience,
                                         start=epoch, max_iter=max_iter, maximize=maximize,
                                         tolerance=tolerance, restore_best=restore_best,
                                         tracker=tracker):
        scheduler.step(val_obj)
        for images, responses in train_loader:
            optimizer.zero_grad()
            loss = full_objective(images.float().cuda(), responses.float().cuda(), model, criterion)
            loss.backward()
            optimizer.step()
        print('Epoch {}, Training loss: {}'.format(epoch, loss))
        optimizer.zero_grad()

    return model, epoch


def run_model_ens(model, gpu_id, criterion, optimizer, scheduler, stop_closure, train_loader,
        epoch, interval, patience, max_iter, maximize, tolerance,
        restore_best, tracker):
    for epoch, val_obj in early_stopping(model, stop_closure,
                                         interval=interval, patience=patience,
                                         start=epoch, max_iter=max_iter, maximize=maximize,
                                         tolerance=tolerance, restore_best=restore_best,
                                         tracker=tracker):
        scheduler.step(val_obj)
        for images, responses in train_loader:
            optimizer.zero_grad()
            loss = full_objective(images.float().to('cuda:{}'.format(gpu_id)), responses.float().to('cuda:{}'.format(gpu_id)), model, criterion)
            loss.backward()
            optimizer.step()
        print('Epoch {}, Training loss: {}'.format(epoch, loss))
        optimizer.zero_grad()

    return model, epoch

def train_model(model, seed, train, val, test, **config):

    tracker = MultipleObjectiveTracker(
        poisson=partial(poisson_stop_mc, model, val),
        gamma=partial(gamma_stop_mc, model, val),
        correlation=partial(corr_stop_mc, model, val),
        exponential=partial(exp_stop_mc, model, val)
                        )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=0.2,
                                                           patience=5,
                                                           threshold=1e-3,
                                                           min_lr=1e-4,
                                                           )

    stop_closure = lambda model: corr_stop_mc(model, val)
    model, epoch = run(model=model,
                       criterion=PoissonLoss(avg=False),
                       optimizer=optimizer,
                       scheduler=scheduler,
                       stop_closure=stop_closure,
                       train_loader=train,
                       epoch=0,
                       interval=1,
                       patience=10,
                       max_iter=config['max_iter'],
                       maximize=True,
                       tolerance=1e-5,
                       restore_best=True,
                       tracker=tracker,
                       )
    tracker.finalize()
    return np.max(tracker.log['correlation']), tracker.log, model.state_dict()


def train_model_ens(model, seed, **config):
    print('started')
    model.to('cuda:{}'.format(config['gpu_id']))
    loaders = create_dataloaders('/notebooks/data/static20892-3-14-preproc0.h5', batch_size=64)
    train, val, test = loaders['train'],loaders['val'], loaders['test']
    tracker = MultipleObjectiveTracker(
        poisson=partial(poisson_stop_mc, model, val, config['gpu_id']),
        correlation=partial(corr_stop_mc, model, val, config['gpu_id']),
                        )
    print('loaders, tracker init for model {}'.format(config['gpu_id']))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    print('optimizer {}'.format(config['gpu_id']))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=0.2,
                                                           patience=5,
                                                           threshold=1e-3,
                                                           min_lr=1e-4,
                                                           )
    print('scheduler {}'.format(config['gpu_id']))

    stop_closure = lambda model: corr_stop_mc(model, val, config['gpu_id'])
    model, epoch = run_model_ens(model=model,
                                 gpu_id=config['gpu_id'],
                                 criterion=PoissonLoss(avg=False),
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 stop_closure=stop_closure,
                                 train_loader=train,
                                 epoch=0,
                                 interval=1,
                                 patience=10,
                                 max_iter=config['max_iter'],
                                 maximize=True,
                                 tolerance=1e-5,
                                 restore_best=True,
                                 tracker=tracker,
                                 )
    tracker.finalize()
    config['manager_list'].append([np.max(tracker.log['correlation']), tracker.log, config['gpu_id']])


def split_training(ens, model_ids, seed, **config):
    print('start split training')
    manager = mp.Manager()
    results = manager.list()
    config['manager_list'] = results
    processes = []
    for gpu_id, model in enumerate(ens.models[model_ids:model_ids + 2]):
        config['gpu_id'] = gpu_id
        p = mp.Process(target=train_model_ens, args=(model, 5,),
                       kwargs=config)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    all_results = [result for result in results]
    return all_results


def train_ensemble(ensemble, seed, **config):
    all_results = []
    for i in range(0, 10, 2):
        res = split_training(ensemble, i, seed, **config)
        all_results.append(res)
    return all_results

