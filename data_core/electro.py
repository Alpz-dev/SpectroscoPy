import string
from data_core.data import *
from data_core.plot import *

def import_electro(file_name, CV_last = False):
    data = open(file_name, "r")

    _x = []
    _y = []

    if CV_last:
        break_flag = False
        for line in reversed(data.read().splitlines()):
            if break_flag:
                break
            else:
                if line != "":
                    if line[0] not in string.ascii_letters:
                        for i in range(len(line)):
                            if line[i] == ",":
                                x = line[0:i]
                                y = line[i + 1::]
                                if float(x) != 1.0:

                                    _x.append(float(x))
                                    _y.append(float(y))
                                    break_flag = False
                                else:
                                    break_flag = True


    else:
        for line in data.read().splitlines():
            if line != "":
                if line[0] not in string.ascii_letters:
                    for i in range(len(line)):
                        if line[i] == ",":
                            x = line[0:i]
                            y = line[i + 1::]
                            _x.append(float(x))
                            _y.append(float(y))

    return Data(np.array(_x), np.array(_y),
                x_label = "Potential (V)",
                y_label = "Current (A)")



