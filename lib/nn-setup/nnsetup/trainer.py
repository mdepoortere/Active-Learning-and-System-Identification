from functools import partial
import numpy as np
import torch
from nnsetup.stop_measures import corr_stop_mc, poisson_stop_mc, gamma_stop_mc, exp_stop_mc, full_objective
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
        running_loss = 0
        for images, responses in train_loader:
            optimizer.zero_grad()
            loss = full_objective(images.float().cuda(), responses.float().cuda(), model, criterion)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu().item()
        print('Epoch {}, Training loss: {}'.format(epoch, running_loss/len(train_loader)))
        optimizer.zero_grad()

    return model, epoch


def train_model(model, dataloaders,seed=5, uid=None, cb=None, patience=4, lr=0.001, weight_decay=1e-6, max_iter=100):
    train, val, test = dataloaders['train'], dataloaders['val'], dataloaders['test']
    tracker = MultipleObjectiveTracker(
        poisson=partial(poisson_stop_mc, model, val),
        gamma=partial(gamma_stop_mc, model, val),
        correlation=partial(corr_stop_mc, model, val),
        exponential=partial(exp_stop_mc, model, val)
                        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=0.2,
                                                           patience=patience,
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
                       patience=patience,
                       max_iter=max_iter,
                       maximize=True,
                       tolerance=1e-5,
                       restore_best=True,
                       tracker=tracker,
                       )
    tracker.finalize()
    return np.max(tracker.log['correlation']), tracker.log, model.state_dict()
