from data_core.data import Data
from data_core.plot import *
from data_core.gaussian import *

def import_TDDFT(file_path, width, x_window, resolution = 1, data_label = "", lorenz = False):
    energies = []
    strengths = []

    file = open(file_path)

    for line in file:
        if " Excited State " in line:
            energies.append(float(line.split()[6]))
            strengths.append(float(line.split()[8][2:]))

    bands = []

    for energy, strength in zip(energies, strengths):
        if lorenz:
            bands.append(Lorentzian(strength, energy, width))
        else:
            bands.append(Gaussian(strength, energy, width))


    x = np.array([x_window[0]+(i*resolution) for i in range(0, int((x_window[1]-x_window[0]) * (1/resolution)), 1)])

    datas = []

    for band in bands:
        datas.append(Data(x, band.eval(x), data_label = data_label))

    return np.sum(datas)


l = []
for i in range(10, 20, 10):
    l.append(import_TDDFT("Test_data/test data/out_TDDFT_Ni4_SCH38_DCM_50.log",
                     width = i,
                     x_window = (300, 800),
                     resolution = 0.5,
                     data_label = "Ni6_" + str(i),
                     lorenz = False))


from data_core.fileIO import export_csv
export_csv(l[0], "export_csv test")

multi_plot(np.array(l), fig_num = 2)

plt.show()