import torch
import netCDF4 as nc
import numpy as np


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

    number_float = torch.tensor(np.float(np.int(file[15:22] + file[23:26])))
    tensor_data[:, 0] = number_float

    latitude = torch.tensor(ds['LATITUDE'][:].data)
    tensor_data[:, 1] = latitude

    longitude = torch.tensor(ds['LONGITUDE'][:].data)
    tensor_data[:, 2] = longitude

    date_time = ds['REFERENCE_DATE_TIME'][:].data
    date_time_adjusted = read_date_time(date_time)
    tensor_data[:, 3] = date_time_adjusted

    for i in range(n_intervals):
        tensor_data[i, 4] = (max_pres - min_pres) / n_intervals * (i + 1)

    if 'TEMP' in variab:
        temp = ds['TEMP'][:].data
        pres_temp = ds['PRES_TEMP'][:].data

        for i in range(len(pres_temp)):
            pres_temp_selected = pres_temp[i]
            for j in range(n_intervals - 1):
                if (tensor_data[j, 4] + dim_interval) > pres_temp_selected > tensor_data[j, 4]:
                    tensor_data[j, 5] = float(temp[i])

        tensor_data = tensor_data[tensor_data[:, 5] > 0]
        new_dimension_rows = tensor_data.shape[0]

    if 'PSAL' in variab:
        psal = ds['PSAL'][:].data
        pres_psal = ds['PRES_PSAL'][:].data

        for i in range(len(pres_psal)):
            pres_psal_selected = pres_psal[i]
            for j in range(new_dimension_rows):
                if (tensor_data[j, 4] + dim_interval) > pres_psal_selected > tensor_data[j, 4]:
                    tensor_data[j, 6] = float(psal[i])

        tensor_data = tensor_data[tensor_data[:, 6] > 0]
        new_dimension_rows = tensor_data.shape[0]

    if 'CHLA' in variab:
        chla = ds['CHLA'][:].data
        pres_chla = ds['PRES_CHLA'][:].data

        for i in range(len(pres_chla)):
            pres_chla_selected = pres_chla[i]
            for j in range(new_dimension_rows):
                if (tensor_data[j, 4] + dim_interval) > pres_chla_selected > tensor_data[j, 4]:
                    tensor_data[j, 7] = float(chla[i])
        tensor_data = tensor_data[tensor_data[:, 7] > 0]
        new_dimension_rows = tensor_data.shape[0]

    if 'DOXY' in variab:
        doxy = ds['DOXY'][:].data
        pres_doxy = ds['PRES_DOXY'][:].data

        for i in range(len(pres_doxy)):
            pres_doxy_selected = pres_doxy[i]
            for j in range(new_dimension_rows):
                if (tensor_data[j, 4] + dim_interval) > pres_doxy_selected > tensor_data[j, 4]:
                    tensor_data[j, 8] = float(doxy[i])

        tensor_data = tensor_data[tensor_data[:, 8] > 0]
        new_dimension_rows = tensor_data.shape[0]

    if 'NITRATE' in variab:
        nitrate = ds['NITRATE'][:].data
        pres_nitrate = ds['PRES_NITRATE'][:].data

        for i in range(len(pres_nitrate)):
            pres_nitrate_selected = pres_nitrate[i]
            for j in range(new_dimension_rows):
                if (tensor_data[j, 4] + dim_interval) > pres_nitrate_selected > tensor_data[j, 4]:
                    tensor_data[j, 9] = float(nitrate[i])

        tensor_data = tensor_data[tensor_data[:, 9] > 0]

    return tensor_data
