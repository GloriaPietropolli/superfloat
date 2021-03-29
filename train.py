"""
definition of the train function of the neural network
both the routine with mini-batches and without mini-batches
"""

import torch
from IPython import display
from hyperparameter import *

losses = []

if mb_flag == 1:
    path = path_target + "mb/"
if mb_flag == 0:
    path = path_target + "no_mb/"


def train(model, epoch, data, target, optimizer):
    file = open(path + "epoch" + str(epoch) + "_lr" + str(lr) + "_bs" + str(batch_size) + '_result.txt', "w+")
    for t in range(epoch):
        output = model(data)
        criterion = torch.nn.L1Loss()  # criterion=torch.nn.L1Loss()
        loss = criterion(output, target)
        losses.append(loss)

        # print(f"[MODEL]: {top + 1}, [EPOCH]: {t}, [LOSS]: {loss.item():.6f}")
        print(f"[EPOCH]: {t}, [LOSS]: {loss.item():.6f}")
        file.write(f"[EPOCH]: {t}, [LOSS]: {loss.item():.6f}")
        display.clear_output(wait=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    file.close()


def train_mb(model, epoch, data, target, optimizer, ds):
    file = open(path + "epoch" + str(epoch) + "_lr" + str(lr) + "_bs" + str(batch_size) + '_result.txt', "w+")
    for t in range(epoch):
        for batch_idx, (data, target) in enumerate(ds):
            output = model(data)
            criterion = torch.nn.L1Loss()  # criterion=torch.nn.L1Loss()
            loss = criterion(output, target)
            losses.append(loss)

            # if (batch_idx % 50 == 0) or ((batch_idx * len(data) + batch_size) == len(ds.dataset)):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                t, batch_idx * len(data), len(ds.dataset),
                   100. * batch_idx / len(ds), loss.item()))
            file.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                t, batch_idx * len(data), len(ds.dataset),
                   100. * batch_idx / len(ds), loss.item()))
            display.clear_output(wait=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    file.close()


def mylosses():
    return losses
