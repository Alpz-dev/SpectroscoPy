import csv

import numpy as np

import re
from SpectroscoPy.data import Data
import string


def import_regex(file_name: str, data_label="", x_label="", y_label="", dimension: tuple = (1, 1), encoding = "utf8", split: str = ",") -> Data:
    file = open(file_name, 'r', encoding=encoding)

    text = file.read()
 
    #vals = re.findall("\\d+.*\\d[\,\\t]\\d+.*\\d[\,\\t]\\n?", text)
    # \\-?\\d+(e\\-?\\d+)?\\,?\\.?\\d*[\\,\s]\s?\\-?\\d+(e\\-?\\d+)?\\,?\\.?\\d*\\,?\s?\\n

    #\\-?\\d+[\\de\\-\\.]*?\\d[\\,\s]\s?\\-?\\d+[\\de\\-\\.]*?\\d\\,?\s?\\n WORKS
    number_val_regex = "\\-?\\d+[\\de?E?\\-?\\.?\\,?]"

    vals = re.findall("\\-?\\d+[\\de?E?\\-?\\.?\\,?]*[\\,\s]\s?\\-?\\d+[\\de?E?\\-?\\.?\\,?]*\\n", text)
    print(vals)
    vals = [line.split(split) for line in vals]
    x = []
    y = []
    for val in vals:
        x.append(float(val[0]))
        y.append(float(val[1]))

    return Data(np.array(x), np.array(y), data_label=data_label, x_label=x_label, y_label=y_label)








# Imports CSV (comma separated value) files while ignoring any text or headers
# NOTE: ONLY IMPORTS NUMERICAL DATA (i.e. values comprised of only 0, 1, 2, 3, ... 9)
def import_csv(file_name: str, data_label="", x_label="", y_label="", encoding = "utf16") -> Data:
    file = open(file_name, 'r', encoding=encoding)



    data = csv.reader(file)


    x = []
    y = []
    #std = []
    for line in data:
        if line:
            if line[0]:
                print(line)
                x.append(float(line[0]))
                y.append(float(line[1]))
                #std.append(float(line[2]))

    dat = Data(np.array(x), np.array(y), data_label=data_label, x_label=x_label, y_label=y_label)
    #dat.stds = np.array(std)
    return dat


# Imports *.txt UV-Vis data from the Agilent ChemStation v10.0.1 software for Windows XP
# Creates a data obj containing x and y values from the file


def import_data(file_name: str, data_label="", x_label="", y_label="") -> Data:
    if file_name.endswith(".CSV"):
        return import_csv(file_name, data_label=data_label, x_label=x_label, y_label=y_label)

    data = open(file_name, "r")

    x = []
    y = []

    for line in data.read().splitlines():
        has_alpha = False
        for char in line:
            if char.isalpha():
                has_alpha = True
                break
        if not has_alpha:
            line = line.strip()
            for i in range(len(line)):
                if line[i].isspace():
                    x_val = float(line[:i].replace(',', ''))
                    y_val = float(line[i + 1:].replace(',', ''))
                    x.append(x_val)
                    y.append(y_val)

    return Data(np.array(x), np.array(y), data_label=data_label, x_label=x_label, y_label=y_label)


def export_txt(data: Data, file_path: str, delimiter: str = "\t", x_window: tuple = None):
    file = open(file_path + ".txt", "w")

    start = 0
    end = data.x.size
    if x_window is not None:
        start = data.find_closest(x_window[0])
        end = data.find_closest(x_window[1])

    for i in range(start, end):
        line = f"{data.x[i]}{delimiter}{data.y[i]}\n"
        file.writelines(line)


def export_csv(data: Data, file_path: str, x_window: tuple = None, verbose: bool = False):
    """
    Exports a Data object into a comma separated value file.

    Ex. 1.432543,4.23452

    :param data: the Data to be exported
    :param file_path: the file path to write to including the file name (do not include the file extension)
    :param x_window: the x range of the Data to be exported
    :param verbose: includes metadata into the file (data label, x label, etc.)
    :return: None

    """
    file = open(file_path + ".csv", "w")

    start = 0
    end = data.x.size

    if x_window is not None:
        start = data.find_closest(x_window[0])
        end = data.find_closest(x_window[1])

    if verbose:
        metadata = [data.data_label, data.x_label, data.y_label]
        for val in metadata:
            file.writelines(f"{val}\n")

    for i in range(start, end):
        line = f"{data.x[i]},{data.y[i]}\n"
        file.write(line)
