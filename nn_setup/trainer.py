from functools import partial
import numpy as np
import torch
from nn_setup.stop_measures import corr_stop_mc, poisson_stop_mc, gamma_stop_mc, exp_stop_mc, full_objective

from mlutils.training import early_stopping, MultipleObjectiveTracker
from mlutils.measures import PoissonLoss


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


def run_ensemble(ensemble, model_id, gpu_id, criterion, optimizer, scheduler, stop_closure, train_loader,
        epoch, interval, patience, max_iter, maximize, tolerance,
        restore_best, tracker):
    for epoch, val_obj in early_stopping(ensemble, stop_closure,
                                         interval=interval, patience=patience,
                                         start=epoch, max_iter=max_iter, maximize=maximize,
                                         tolerance=tolerance, restore_best=restore_best,
                                         tracker=tracker):
        scheduler.step(val_obj)
        for images, responses in train_loader:
            optimizer.zero_grad()
            loss = full_objective(images.float().to(gpu_id),
                                  responses[model_id, :, :].to(gpu_id), ensemble.models[model_id].to(gpu_id),
                                  criterion)
            loss.backward()
            optimizer.step()
        print('Epoch {}, Training loss: {}'.format(epoch, loss))
        optimizer.zero_grad()

    return ensemble, epoch


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


def train_ensemble(ensemble, seed, train, val, test, **config):

    tracker = MultipleObjectiveTracker(
        poisson=partial(poisson_stop_mc, ensemble, val),
        gamma=partial(gamma_stop_mc, ensemble, val),
        correlation=partial(corr_stop_mc, ensemble, val),
        exponential=partial(exp_stop_mc, ensemble, val)
                        )

    optimizer = torch.optim.Adam(ensemble.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=0.2,
                                                           patience=5,
                                                           threshold=1e-3,
                                                           min_lr=1e-4,
                                                           )

    stop_closure = lambda model: corr_stop_mc(model, val)
    for i in range(ensemble.n_models):
        ensemble, epoch = run_ensemble(ensemble=ensemble,
                                       model_id=i,
                                       gpu_id='cuda:0',
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
    return np.max(tracker.log['correlation']), tracker.log, ensemble.state_dict()
