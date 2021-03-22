import pandas as pd
from data_single_station import *

path = 'data/6901765/MR6901765_001.nc'
ds = nc.Dataset(path)

data = pd.read_csv('data/Float_Index.txt', header=None)
data = data.to_numpy()
data = data[:, 0]
list_nc_data = []
for i in data:
    list_nc_data.append(i)

vars = []
for var in ds.variables:
    vars.append(var)


dimension_matrix_single_emersion = 1000
input_dimension = 10
size_matrix_info = [dimension_matrix_single_emersion, input_dimension]

min_pres = 0
max_pres = 1000
n_intervals = 100
pressure_info = [min_pres, max_pres, n_intervals]
dim_interval = (max_pres - min_pres) / n_intervals

data_single_emersion(path, size_matrix_info, pressure_info)


def group_data_same_station():
    file_first_station = []
    first_station = list_nc_data[0][0:7]
    for file in list_nc_data:
        if file[0:7] == first_station:
            file_first_station.append(file)
    return file_first_station


def data_single_station():
    file_first_station = group_data_same_station()
    tensor_data = data_single_emersion('data/' + file_first_station[0], size_matrix_info, pressure_info)
    file_first_station.remove(file_first_station[0])
    for file_selected in file_first_station:
        data_to_add = data_single_emersion('data/' + file_selected, size_matrix_info, pressure_info)
        if data_to_add is not None:
            tensor_data = torch.cat((tensor_data, data_to_add), 0)
    return tensor_data


data_single_float = data_single_station()
