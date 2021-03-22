import torch
import netCDF4 as nc
import numpy as np

dict_variables = {'LATITUDE': 1, 'LONGITUDE': 2, 'REFERENCE_DATE_TIME': 3, 'TEMP': 5, 'PSAL': 6, 'CHLA': 7, 'DOXY': 8,
                  'NITRATE': 9}

list_fix_var = ['LATITUDE', 'LONGITUDE', 'REFERENCE_DATE_TIME']
list_nonfix_var = dict_variables.keys() - list_fix_var


def read_date_time(date_time):
    date_time_decoded = ''
    for i in range(0, 14):
        new_digit = date_time[i].decode('UTF-8')
        date_time_decoded += new_digit
    date_time_decoded = np.int(date_time_decoded)
    return date_time_decoded


def data_single_emersion(file, size_matrix_info, pressure_info):
    ds = nc.Dataset(file)
    variab = []
    for var in ds.variables:
        variab.append(var)

    dimension_matrix_single_emersion, input_dimension = size_matrix_info
    min_pres, max_pres, n_intervals = pressure_info
    dim_interval = (max_pres - min_pres) / n_intervals

    tensor_data = torch.zeros(dimension_matrix_single_emersion, input_dimension)

    number_float = torch.tensor(np.float(file[5:12]))
    tensor_data[:, 0] = number_float

    for var in list_fix_var:
        if var == 'REFERENCE_DATE_TIME':
            date_time = ds[var][:].data
            index = dict_variables[var]
            date_time_adjusted = read_date_time(date_time)
            tensor_data[:, index] = date_time_adjusted
        else:
            var_value = torch.tensor(ds[var][:].data)
            index = dict_variables[var]
            tensor_data[:, index] = var_value

    for i in range(n_intervals):
        tensor_data[i, 4] = (max_pres - min_pres) / n_intervals * (i + 1)

    for var in list_nonfix_var:
        if var in variab:
            new_dimension_rows = tensor_data.shape[0]
            var_values = ds[var][:].data
            pres_var_values = ds['PRES_' + var][:].data
            index = dict_variables[var]

            if var == 'CHLA':
                size_range = n_intervals - 1
            else:
                size_range = new_dimension_rows

            for i in range(len(pres_var_values)):
                pres_var_selected = pres_var_values[i]
                for j in range(size_range):
                    if (tensor_data[j, 4] + dim_interval) > pres_var_selected > tensor_data[j, 4]:
                        tensor_data[j, index] = float(var_values[i])

            tensor_data = tensor_data[tensor_data[:, index] > 0]

    tensor_data = tensor_data[tensor_data[:, 4] > 0]
    return tensor_data
