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
input_dimension = 11
size_matrix_info = [dimension_matrix_single_emersion, input_dimension]

min_pres = 0
max_pres = 1000
n_intervals = 100
pressure_info = [min_pres, max_pres, n_intervals]
dim_interval = (max_pres - min_pres) / n_intervals


def float_number():  # out1: list of all the float number; out2= list of all measurements per float number
    float_number = []
    measurement_per_float = []
    while list_nc_data:
        considered_station = list_nc_data[0][0:7]
        list_measurement_considered_float = []
        for file in list_nc_data:
            if file[0:7] == considered_station:
                list_measurement_considered_float.append(file)
            list_nc_data.remove(file)
        float_number.append(considered_station)
        measurement_per_float.append(list_measurement_considered_float)

    return float_number


list_float_number = float_number()
list_float_number = list(dict.fromkeys(list_float_number))

list_nc_data = []
for i in data:
    list_nc_data.append(i)


def extract_emersion_per_float(float_index):
    result_this_float = []
    for j in range(np.size(list_nc_data)):
        if list_float_number[float_index] == list_nc_data[j][0:7]:
            result_this_float.append(list_nc_data[j])
    return result_this_float


def data_single_station(station_considered):  # station considered is indexing with the number of the position in the
    # float_number list
    file_station = extract_emersion_per_float(station_considered)
    tensor_data = data_single_emersion('data/' + file_station[0], size_matrix_info, pressure_info)
    file_station.remove(file_station[0])
    for file_selected in file_station:
        data_to_add = data_single_emersion('data/' + file_selected, size_matrix_info, pressure_info)
        if data_to_add is not None:
            tensor_data = torch.cat((tensor_data, data_to_add), 0)
    return tensor_data


list_tensor_data = []


def data_all_station():
    for index_station in range(len(list_float_number)):
        station_considered = list_float_number[index_station]
        tensor_considered = data_single_station(index_station)
        list_tensor_data.append(tensor_considered)
        np.savetxt('data_elabored/data_station_' + str(station_considered) + '.csv', tensor_considered, delimiter=',')
        print('New tensor saved!')
