
import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import scipy as sp
import os
plt.style.use('ggplot')

dic, data = ng.fileio.bruker.read('Test_data/Ni-CyHT_10/Ni-CyHT/10')
lst = []
fid = ng.bruker.remove_digital_filter(dic, data)
fid = ng.proc_base.zf_size(fid, 32768) # <2>
fid = ng.proc_base.rev(fid) # <3>
fid = ng.proc_base.fft(fid)
lst.append(fid)
lst = np.array(lst)


def plotspectra(ppms, data, start=None, stop=None):

    if start: # <1>
        ixs = list(ppms).index(start)
        ppms = ppms[ixs:]
        data = data[:,ixs:]
    if stop:
        ixs = list(ppms).index(stop)
        ppms = ppms[:ixs]
        data = data[:,:ixs]


    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1,1,1)
    for n in range(data.shape[0]):
        ax.plot(ppms, data[n,:])

    ax.set_xlabel('ppm')
    ax.invert_xaxis()

    return fig

ppms = np.arange(0, fid.shape[0])
fig = plotspectra(ppms, lst)

plt.show()

plt.show()




