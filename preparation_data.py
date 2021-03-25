import pandas as pd
import torch
from aggregation_data import list_float_total
from preparation_function import preparation_routine


path = 'data_elabored/'

for float_number in list_float_total:
    list_of_pandas = []
    list_of_tensor = []
    tensor_to_add = pd.read_csv(path + 'data_station_' + str(float_number) + '.csv')
    tensor_to_add.columns = ['float_id', 'number', 'lat', 'lon', 'date_time', 'pres', 'temp', 'psal', 'chla', 'doxy',
                             'nitrate', 'BBP700', 'downwelling_par']
    list_of_pandas.append(tensor_to_add)
    tensor_to_add = torch.tensor(tensor_to_add.values)


def aggregation_all_float_info():
    aggregation_tensor = list_of_tensor[0]
    list_of_tensor.remove(aggregation_tensor)
    for tensor in list_of_tensor:
        aggregation_tensor = torch.cat((aggregation_tensor, tensor), 0)
    return aggregation_tensor


dataset = aggregation_all_float_info()
dataset = dataset[torch.randperm(dataset.size()[0])]  # shuffle of the samples
dataset_size = len(dataset[:, 0])

percentage_samples_for_training = 80  # 80% of samples are used for training
training_set_size = int(dataset_size * percentage_samples_for_training/100)
validation_set_size = dataset_size - training_set_size

training_set, validation_set = torch.utils.data.random_split(dataset, [training_set_size, validation_set_size])
training_set, validation_set = training_set.float(), validation_set.float()

training_set, validation_set = preparation_routine(training_set, training_set), preparation_routine(validation_set,
                                                                                                    training_set)






