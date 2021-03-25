"""
Python file containing all the functions necessary to preprocessing the data in order to feed correctly the nn
"""

import torch
import numpy as np


def fix_datetime(dataset):  # computation of the decimal year
    for i in range(len(dataset[:, 0])):  # iteration on the rows (i.e. the samples)
        date_time = str(dataset[i, 4].item())
        year, month, day = date_time[0:4], date_time[4:6], date_time[6:8]
        hour, min = date_time[8:10], date_time[10:12]
        dataset[i, 0] = int(year) + float(month / 10)  # fix data input into decimal year
    return dataset


def fix_latitude(dataset):  # divide latitude input for 90
    dataset[:, 2] = dataset[:, 2] / 90
    return dataset


def fix_longitude(dataset):
    first_half = dataset[:, 0:4]
    second_half = dataset[:, 4:]
    dataset[:, 3] = np.abs(1 - np.mod(dataset[:, 2] - 110, 360) / 180)  # fix longitude input
    new_column = np.abs(1 - np.mod(dataset[:, 3] - 20, 360) / 180)  # fix longitude input
    dataset = torch.cat((first_half, new_column, second_half), 1)
    return dataset


def fix_pressure(dataset):
    dataset[:, 4] = dataset[:, 4] / 20000 + (1 / ((1 + np.exp(-dataset[:, 4] / 300)) ** 3))  # fix pressure input
    return dataset


def get_mean_std_from_training_data(dataset):
    mean = [dataset[:, i].mean() for i in range(dataset.size()[1])]  # mean of the columns
    std = [dataset[:, i].std() for i in range(dataset.size()[1])]  # std of the columns
    return mean, std


def normalization(dataset, training):  # we want to normalize the dataset using mean and std of training set
    mean, std = get_mean_std_from_training_data(training)
    for i in range(dataset.size()[1]):  # iterations over the columns
        dataset[:, i] = 2 / 3 * (dataset[:, i] - mean[i]) / std[i]
    return dataset


def preparation_routine(dataset, training):
    dataset = fix_datetime(dataset)
    dataset = fix_latitude(dataset)
    dataset = fix_longitude(dataset)
    dataset = fix_pressure(dataset)
    dataset = normalization(dataset, training)
    return dataset
