import string
from SpectroscoPy.data import *
from SpectroscoPy.plot import *

def import_electro(file_name, CV_last = False, data_label = ""):
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
                    print(line)
                    for i in range(len(line)):
                        if line[i] == ",":
                            x = line[0:i]
                            y = line[i + 1::]
                            _x.append(float(x))
                            _y.append(float(y))

    return Data(np.array(_x), np.array(_y), data_label = data_label)


def import_CA(file_name, data_label = ""):
    data = open(file_name, "r")

    _x = []
    _y = []

    for line in data.read().splitlines():
        if line != "":
            if line[0] not in string.ascii_letters:
                vals = line.split(sep=", ")
                x = vals[0]
                y = vals[2]
                _x.append(float(x))
                _y.append(float(y))

    return Data(np.array(_x), np.array(_y), data_label = data_label)


