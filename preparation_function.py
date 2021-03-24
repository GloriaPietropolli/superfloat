
def fix_datetime(dataset):  # computation of the decimal year
    for i in range(len(dataset[:, 0])):  # iteration on the rows (i.e. the samples)
        date_time = str(dataset[i ,4].item())
        year, month, day = date_time[0:4], date_time[4:6], date_time[6:8]
        hour, min = date_time[8:10], date_time[10:12]
        dataset[i, 0] = int(year) + float(month/10)  # fix data input into decimal year


