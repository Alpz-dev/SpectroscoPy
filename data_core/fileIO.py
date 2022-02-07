


def export_tabtxt(data, file_path, x_window=None, verbose = False):
    file = open(file_path + ".txt", "w")

    start = 0
    end = data.x.size

    if x_window is not None:
        start = data.find_closest(x_window[0])
        end = data.find_closest(x_window[1])

    for i in range(start, end):
        line = str(data.x[i]) + "\t" + str(data.y[i]) + "\n"
        file.writelines(line)


def export_csv(data, file_path, x_window: tuple = None, verbose=False):
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

