from data_core.data import *


def export_txt(data, file_name, x_window = None):
    file = open(file_name + ".txt", "w")

    start = 0
    end = data.x.size

    if x_window is not None:
        start = data.find_closest(x_window[0])
        end = data.find_closest(x_window[1])

    for i in range(start, end):
        line = str(data.x[i]) + "\t" + str(data.y[i]) + "\n"
        file.writelines(line)