"""
Python file containing the routine to preprocessing the data in order to feed correctly the nn
"""

import pandas as pd
import torch
from preparation_function import preparation_routine, split_data_target, normalization, normalization_target
from hyperparameter import index_target, list_float_total

path = 'data_elabored/'

list_of_pandas = []
list_of_tensor = []
for float_number in list_float_total:
    pandas_to_add = pd.read_csv(path + 'data_station_' + str(float_number) + '.csv')
    pandas_to_add.columns = ['float_id', 'number', 'lat', 'lon', 'date_time', 'pres', 'temp', 'psal', 'chla', 'doxy',
                             'nitrate', 'BBP700', 'downwelling_par']
    list_of_pandas.append(pandas_to_add)
    tensor_to_add = torch.tensor(pandas_to_add.values)
    list_of_tensor.append(tensor_to_add)


def aggregation_all_float_info():
    aggregation_tensor = list_of_tensor[0]
    for tensor in list_of_tensor:
        aggregation_tensor = torch.cat((aggregation_tensor, tensor), 0)
    return aggregation_tensor


dataset = aggregation_all_float_info()
dataset = dataset[torch.randperm(dataset.size()[0])]  # shuffle of the samples

dataset = preparation_routine(dataset)

dataset_input, dataset_output = split_data_target(dataset, index_target)
dataset_size = len(dataset_input[:, 0])
input_size = len(dataset_input[0, :])

percentage_samples_for_training = 80  # 80% of samples are used for training
training_set_size = int(dataset_size * percentage_samples_for_training / 100)
validation_set_size = dataset_size - training_set_size

training_input, validation_input = dataset_input[0:training_set_size, :], dataset_input[training_set_size:, :]
training_input, validation_input = training_input.float(), validation_input.float()

training_input = normalization(training_input, training_input)
validation_input = normalization(validation_input, training_input)

training_target, validation_target = dataset_output[0:training_set_size], dataset_output[training_set_size:]
training_target, validation_target = training_target.float(), validation_target.float()

training_target = normalization_target(training_target, training_target)
validation_target = validation_target(validation_target, training_target)
