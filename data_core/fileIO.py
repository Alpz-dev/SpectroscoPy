# Imports CSV (comma separated value) files while ignoring any text or headers
# NOTE: ONLY IMPORTS NUMERICAL DATA (i.e. values comprised of only 0, 1, 2, 3, ... 9)
def import_csv(file_name, data_label=""):
    file = open(file_name, 'r')
    data = csv.reader((line.replace('\0', '') for line in file), delimiter=",")
    x = np.zeros(0)
    y = np.zeros(0)
    for line in data:
        if line != []:
            if line[0].isnumeric():
                x.append(float(line[0]))
                y.append(float(line[1]))
    return Data(np.array(x), np.array(y), data_label=data_label)


# Imports *.txt UV-Vis data from the Agilent ChemStation v10.0.1 software for Windows XP
# Creates a data obj containing x and y values from the file
def import_data(file_name, instr="", data_label="", harlem_data=False, harlem_mc_data=False, harlem_scan_data=False,
                harlem_enter_data=False, decimate=None):
    if harlem_data:
        data = open(file_name, "r")
        y = []
        x = []
        i = 0
        for line in data.read().splitlines():
            i += 1
            y.append(float(line))
            x.append(i)
        return Data(x, y, file_name=file_name, instrument=instr)

    if harlem_enter_data:
        data = open(file_name, "r")
        y = []
        x = []
        lines = data.read().splitlines()
        for i in range(len(lines)):

            if (i - 2) % 3 == 0:
                if i >= 0:

                    if lines[i - 2].strip() != "EPtot":
                        if lines[i - 2].strip() != "NSTEP":
                            if lines[i - 2].strip() != "Etot":
                                if lines[i - 2].strip() != "":
                                    if len(lines[i - 2].strip()) > 5:
                                        y.append(float(lines[i - 2].strip()))
            elif (i + 1) % 3 == 2:
                if lines[i].strip() != "TIME(PS)":
                    if lines[i].strip() != "VOLUME":
                        if float(lines[i].strip()) != 99999:
                            x.append(float(lines[i].strip()))
        return Data(x, y, file_name=file_name, instrument=instr)

    if harlem_mc_data:
        data = open(file_name, "r")
        y = []
        x = []
        n = []
        i = 0
        for line in data.read().splitlines():
            i += 1
            if decimate == None:
                vals = line.split(" ")
                n.append(float(vals[0]))  # nstep
                x.append(float(vals[1]))  # angle
                y.append(float(vals[2]))  # pot. ene.
            else:
                if i % decimate == 0:
                    vals = line.split(" ")
                    n.append(float(vals[0]))  # nstep
                    x.append(float(vals[1]))  # angle
                    y.append(float(vals[2]))  # pot. ene.
        return Data(x, y, n=n, file_name=file_name, instrument=instr)

    if file_name.endswith(".CSV"):
        return import_csv(file_name, data_label=data_label)
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
    return Data(np.array(x), np.array(y), data_label=data_label)


def export_tabtxt(data, file_path, x_window=None):
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

