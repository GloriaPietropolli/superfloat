import pandas as pd
from data_single_station import *

path = 'data/6901765/MR6901765_001.nc'
ds = nc.Dataset(path)

data = pd.read_csv('data/Float_Index.txt', header=None)
data = data.to_numpy()
data = data[:, 0]

vars = []
for var in ds.variables:
    vars.append(var)

dimension_matrix_single_emersion = 1000
input_dimension = 13
size_matrix_info = [dimension_matrix_single_emersion, input_dimension]

min_pres = 0
max_pres = 1000
n_intervals = 100
pressure_info = [min_pres, max_pres, n_intervals]
dim_interval = (max_pres - min_pres) / n_intervals

list_float_total = ['6902875', '6901765', '6901772', '6901775', '6901770', '6902874', '6902965', '6903247', '6901649',
                    '6900807', '6902899', '6901862', '6902873', '6901466', '7900592', '6901476', '6901032', '6902733',
                    '6902898', '6901464', '6903240', '6903262', '6902872', '6901528', '6903237', '6901510', '6901513',
                    '6902904', '6901769', '6902937', '6902879', '6901764', '6901774', '6902903', '6902936', '6901605',
                    '6902969', '6903263', '6903765', '6902688', '6901600', '6901648', '6903250', '6901866', '6903238',
                    '6903249', '6902826', '6901861', '6901653', '6901864', '6902687', '6901465', '7900591', '6902876',
                    '6901460', '6901463', '6902700', '6901483', '6901496', '6901512', '6901768', '6901491', '6901766',
                    '6901773', '6901776', '6902732', '6901771', '6903781', '6902954', '6901596', '6901865', '6901657',
                    '6902870', '6902804', '6902902', '6901860', '6901863', '6901655', '6902935', '7900562', '6901467',
                    '6902803', '6902968', '6901487', '6902901', '6901462', '6901529', '6903235', '6903246', '6903268',
                    '6902828', '6901511', '6903266', '6901767', '6901470', '6902900', '6901490', '6903197']

list_float_number = ['6902969', '6903263', '6903765', '6902688', '6901600', '6901648', '6903250', '6901866', '6903238',
                     '6903249', '6902826', '6901861', '6901653', '6901864', '6902687', '6901465', '7900591', '6902876',
                     '6901460', '6901463', '6902700', '6901483', '6901496', '6901512', '6901768', '6901491', '6901766',
                     '6901773', '6901776', '6902732', '6901771', '6903781', '6902954', '6901596', '6901865', '6901657',
                     '6902870', '6902804', '6902902', '6901860', '6901863', '6901655', '6902935', '7900562', '6901467',
                     '6902803', '6902968', '6901487', '6902901', '6901462', '6901529', '6903235', '6903246', '6903268',
                     '6902828', '6901511', '6903266', '6901767', '6901470', '6902900', '6901490', '6903197']

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

        for i in range(tensor_considered.shape[1]):
            tensor_considered[0, i] = i

        np.savetxt('data_elabored/data_station_' + str(station_considered) + '.csv', tensor_considered, delimiter=',')
        print('New tensor saved : ' + str(station_considered))
