import csv
import sys

import numpy as np
from scipy import signal
from scipy.optimize import *
from SpectroscoPy.gaussian import *
from SpectroscoPy.plot import *
from tqdm import tqdm
import random
from SpectroscoPy.line import *
import pickle
import os
from SpectroscoPy.exponential import *

class Fit(object):
    def __init__(self):
        self.subfunc_data = []
        self.subfuncs = []
        self.fit_data = None
        self.residual_data = None
        self.result = ""
    def __hash__(self):
        return hash(hash(self.fit_data) + hash(len(self.bands_data)))

# Generic useful functions
def file_empty(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return False
    return True


# Definition of the main "data" object class which holds 2D data (i.e. (x, y)) and provides functions to manipulate
class Data(object):
    def __init__(self, x: np.ndarray, y: np.ndarray,
                 x_label: str = 'x',
                 y_label: str = 'y',
                 data_label: str = " ",
                 path: str = ""):

        self.x = x
        self.y = y

        self.x_label = x_label
        self.y_label = y_label

        self.data_label = data_label

        self.x_order = 2
        self.y_order = 0

        self.path = path

        self.stds = np.ones(self.x.size)

    def __add__(self, other):
        if isinstance(other, Data):
            # print(self.y.size, other.y.size)
            # if self.y.size > other.y.size:
            #     other.y = np.append(other.y, np.zeros(self.y.size - other.y.size))
            #     other.x = np.append(other.x, self.x[])
            # else:
            #     self.y = np.append(self.y, np.zeros(other.y.size - self.y.size))
            # print(self.y.size, other.y.size)
            return Data(self.x, self.y + other.y,
                        x_label=self.x_label,
                        y_label=self.y_label,
                        data_label=self.data_label)
        else:
            return self

    def __sub__(self, other):
        if isinstance(other, Data):
            return Data(self.x, self.y - other.y,
                        x_label=self.x_label,
                        y_label=self.y_label,
                        data_label=self.data_label)
        else:
            return self

    def __mul__(self, other):
        if isinstance(other, Data):
            return Data(self.x, self.y * other.y,
                        x_label=self.x_label,
                        y_label=self.y_label,
                        data_label=self.data_label)
        elif isinstance(other, float):
            return Data(self.x, self.y * other,
                        x_label=self.x_label,
                        y_label=self.y_label,
                        data_label=self.data_label)
        else:
            return self

    def __truediv__(self, other):
        if isinstance(other, Data):
            return Data(self.x, np.true_divide(self.y, other.y),
                        x_label=self.x_label,
                        y_label=self.y_label,
                        data_label=self.data_label)
        elif (isinstance(other, float) or isinstance(other, int)):
            return Data(self.x, np.true_divide(self.y, other),
                        x_label=self.x_label,
                        y_label=self.y_label,
                        data_label=self.data_label)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, e):
        return Data(self.x, self.y ** e,
                    x_label=self.x_label,
                    y_label=self.y_label,
                    data_label=self.data_label)

    def __abs__(self):
        return Data(self.x, np.abs(self.y),
                    x_label=self.x_label,
                    y_label=self.y_label,
                    data_label=self.data_label)

    def __repr__(self):
        out = F"{self.x_label}: {self.x}\n" \
              F"{self.y_label}: {self.y}"
        return out

    def __hash__(self):
        return hash(hash(tuple(self.x)) + hash(tuple(self.y)))

    def sqrt(self):
        return Data(self.x, np.sqrt(self.y),
                    x_label=self.x_label,
                    y_label=self.y_label,
                    data_label=self.data_label)

    # returns log base 10 of the y dataset
    def log10(self):
        return Data(self.x, np.log10(self.y),
                    x_label=self.x_label,
                    y_label=self.y_label,
                    data_label=self.data_label)

    # returns the transposition of the data set (i.e. x->y and y->x)
    def transpose(self):
        return Data(self.y, self.x,
                    x_label=self.y_label,
                    y_label=self.x_label,
                    data_label=self.data_label)

    # Returns the index of the array (x or y) that contains a value closest to the target
    def find_closest(self, target, x: bool = True, y: bool = False):
        if x:
            distance = abs(self.x - target)
            return list(distance).index(min(distance))
        if y:
            distance = abs(self.y - target)
            return list(abs(self.y - target)).index(min(distance))

    # Converts nm to ev with Jacobian correction to intensity
    def nm2ev(self):
        _x = np.zeros(self.x.size)
        _y = np.zeros(self.y.size)

        def f(x, y):
            return 1239.84193 / x, y * 1239.84193 / ((1239.84193 / x) ** 2)

        for i in range(self.x.size):
            _x[i], _y[i] = f(self.x[i], self.y[i])

        return Data(np.flip(_x), np.flip(_y),
                    x_label=self.x_label,
                    y_label=self.y_label,
                    data_label=self.data_label)

    # Converts nm to cm^-1
    def nm2cm_inv(self):

        return Data((10 ** 7) / self.x, self.y,
                    x_label=self.x_label,
                    y_label=self.y_label,
                    data_label=self.data_label)

    # Translate dataset on x and/or y within some x_window (defined relative to the x dataset)
    def translate(self,
                  dx: float = 0.0,
                  dy: float = 0.0,
                  x_window: tuple = None):

        start = 0
        end = self.x.size

        if x_window is not None:
            start = int(self.find_closest(x_window[0]))
            end = int(self.find_closest(x_window[1]))

        _x = np.append(self.x[0:start], np.append(self.x[start:end] + dx, self.x[end + 1:-1]))
        _y = np.append(self.y[0:start], np.append(self.y[start:end] + dy, self.y[end + 1:-1]))

        return Data(_x, _y,
                    x_label=self.x_label,
                    y_label=self.y_label,
                    data_label=self.data_label)

    # Crops a data set to a specified x_window
    # NOTE: RETURNED DATASET < ORIGINAL DATASET IN X DIMENSION
    def slice(self, x_window):
        start = 0
        end = self.x.size

        if x_window is not None:
            start = int(self.find_closest(x_window[0]))
            end = int(self.find_closest(x_window[1]))

        _x = self.x[start:end]
        _y = self.y[start:end]

        return Data(_x, _y,
                    x_label=self.x_label,
                    y_label=self.y_label,
                    data_label=self.data_label)
    
    def flip(self, x = True, y = False):
        _x = self.x
        _y = self.y
        if x:
            _x = np.flip(self.x)
        if y:
            _y = np.flip(self.y)

        return Data(_x, _y,
                    x_label=self.x_label,
                    y_label=self.y_label,
                    data_label=self.data_label)

    def range(self, range: tuple):
        prev_range = (np.max(self.x) - np.min(self.x))  
        new_range = max(range) - min(range)

        _x = ((self.x - np.min(self.x) * new_range) / prev_range) + min(range)

        return Data(_x, self.y,
                    x_label=self.x_label,
                    y_label=self.y_label,
                    data_label=self.data_label)
        

    def moving_average(self, window:int):
        _y = []

        for i in range(self.y.size):
            if i < (self.y.size - window):
                _y.append(np.average(self.y[i:i+window]))
        _x = self.x[0:self.x.size-window]
        return Data(_x, np.array(_y),
                    x_label=self.x_label,
                    y_label=self.y_label,
                    data_label=self.data_label)



    def randomize(self, std: float):
        """
        Randomizes y values centered the original value. Uses a gaussian distribution with a mean of the original value
        and a specified standard deviation.

        :param std: standard deviation of the gaussian distribution
        :return: randomized Data

        """
        y = []
        for val in self.y:
            y.append(random.gauss(val, std))
        self.y = np.array(y)
        return Data(self.x, y,
                    x_label=self.x_label,
                    y_label=self.y_label,
                    data_label=self.data_label)

    # Scales dataset on x and/or y
    def scale(self,
              sx: float = 1.0,
              sy: float = 1.0):

        return Data(self.x * sx, self.y * sy,
                    x_label=self.x_label,
                    y_label=self.y_label,
                    data_label=self.data_label)
    # Normalizes dataset within some x_window (defined relative to the x dataset)
    def normalize(self,
                  x_window: tuple = None):

        start = 0
        end = self.x.size

        if x_window is not None:
            start = int(self.find_closest(x_window[0]))
            end = int(self.find_closest(x_window[1]))

        _max = np.ndarray.max(self.y[start:end])
        _min = np.ndarray.min(self.y[start:end])
        _range = (_max - _min)

        return self.translate(dy = -_min).scale(sy = 1/_range)

    # Returns slope between two points (relative to the x array)
    def slope(self, x1: int, x2: int):
        return (self.y[x2] - self.y[x1]) / (self.x[x2] - self.x[x1])

    # Filters the dataset using a Savitzky-Golay filter implemented in the scipy.signal package
    def filter(self, window: int, poly_order: int,
               deriv: int = 0,
               mode: str = 'interp'):

        return Data(self.x, signal.savgol_filter(self.y, window, poly_order, deriv=deriv, mode=mode),
                    x_label=self.x_label,
                    y_label=self.y_label,
                    data_label=self.data_label)

    # Takes derivative of dataset
    # order specifies number of derivatives to be taken
    def differentiate(self,
                      order: int = 1,
                      x_window=None):

        start = 0
        end = self.x.size

        if x_window is not None:
            start = int(self.find_closest(x_window[0]))
            end = int(self.find_closest(x_window[1]))

        def slope(data, x1: int, x2: int):
            return (data.y[x2] - data.y[x1]) / (data.x[x2] - data.x[x1])

        _y = np.copy(self.y[start:end])

        _data = Data(self.x[start:end], _y,
                     x_label=self.x_label,
                     y_label=self.y_label,
                     data_label=self.data_label)

        for o in range(order):

            for i in range(self.y.size):

                if i == self.y.size - 1:
                    _data.y[i] = slope(_data, i - 1, i)
                else:
                    _data.y[i] = slope(_data, i, i + 1)

        return _data

    # Evaluates trapezoidal area under two points (relative to the x array)
    def trapezoidal_area(self, x1: int, x2: int):
        return (self.y[x1] + self.y[x2]) * (self.x[x2] - self.x[x1]) * 0.5

    # Calculates area under the curve between two points (relative to the x array) using the trapezoidal rule
    def area(self, x_window = None):
        start = 0
        end = self.x.size

        if x_window is not None:
            start = int(self.find_closest(x_window[0]))
            end = int(self.find_closest(x_window[1]))


        a = 0

        for i in range(start, end, 1):
            if i == self.x.size - 1:
                a += self.trapezoidal_area(i, i)
            else:
                a += self.trapezoidal_area(i, i + 1)

        return a

    def integrate(self):
        y = []
        for i in range(self.x.size):
            y.append(np.sum(self.y[0:i]))
        return Data(self.x, np.array(y),
                    x_label=self.x_label,
                    y_label=self.y_label,
                    data_label=self.data_label)
    





    # Finds the maximum value within some search window relative to a center point
    # WIP
    def window_maxima_search(self, center: int, window: int,
                             fast: bool = False):
        if not fast:

            _max = (self.y[center], center)

            for i in range(center - window // 2, center + window // 2, 1):
                if (self.y.size - 1) >= i >= 0:
                    if self.y[i] > _max[0]:
                        _max = (self.y[i], i)

            return _max[1]

        else:
            print("Error: Gradient Ascent search is currently WIP. Use fast = False for now.")

    # Finds peak of dataset within some x_window (defined relative to the x dataset)
    # slope_threshold defines the minimum slope required for a peak to be accepted (filters on peak width)
    # amp_threshold defines the minimum amplitude for a peak to be accepted (filters on peak height)
    # amp_threshold is the scalar multiplied by the maximum value in the search range to produce the amp_threshold
    def find_peaks(self,
                   x_window: tuple = None,
                   slope_threshold: float = -0.00001,
                   amp_threshold: float = 0.1,
                   _window: int = None):

        _first_deriv = self.filter(21, 2, deriv=1)

        start = 0
        end = self.x.size

        if x_window is not None:
            start = int(self.find_closest(x_window[0]))
            end = int(self.find_closest(x_window[1]))

        window = 10 ** (self.x_order) * 0.1

        if _window is not None:
            window = _window

        window = int(window)

        result = []

        for i in range(start, end - 1, 1):
            if _first_deriv.y[i] > 0 and _first_deriv.y[i + 1] < 0:
                if _first_deriv.slope(i, i + 1) <= slope_threshold:
                    possible_max = self.window_maxima_search(i, window)
                    if self.y[possible_max] >= amp_threshold * np.max(self.y[start:end]):
                        result.append(self.x[possible_max])

        return np.array(result)

    # Finds shoulders and some peaks within some x_window (defined relative to the x dataset)
    # Uses peak finding on second derivative to locate shoulders
    def find_shoulders(self,
                       x_window: tuple = None,
                       slope_threshold: float = -0.00001,
                       amp_threshold: float = 0.1,
                       _window: int = None):

        _second_deriv = self.filter(5, 2, deriv=2).scale(sy=-1)

        _possible_shoulders = _second_deriv.find_peaks(x_window=x_window,
                                                       slope_threshold=slope_threshold,
                                                       amp_threshold=amp_threshold,
                                                       _window=_window)

        return _possible_shoulders

    # Finds both peaks and shoulders by running both find_peaks and find_shoulders and merging the two results
    # within some convergence threshold where values within the threshold relative to the find_peaks result are combined
    def find_peaks_and_shoulders(self,
                                 x_window: tuple = None,
                                 slope_threshold: float = -0.00001,
                                 amp_threshold: float = 0.1,
                                 _window: int = None,
                                 _conv_threshold: int = None):

        conv_threshold = 3

        if conv_threshold is not None:
            conv_threshold = _conv_threshold

        _peaks = self.find_peaks(x_window=x_window,
                                 slope_threshold=slope_threshold,
                                 amp_threshold=amp_threshold,
                                 _window=_window)

        _shoulders = self.find_shoulders(x_window=x_window,
                                         slope_threshold=slope_threshold,
                                         amp_threshold=amp_threshold,
                                         _window=_window)

        result = list(_peaks)

        conv_threshold = 3

        for shoulder in _shoulders:
            acceptable = []
            for peak in _peaks:
                if abs(peak - shoulder) > conv_threshold:
                    acceptable.append(True)
                else:
                    acceptable.append(False)
                    break
            if np.product(acceptable):
                result.append(shoulder)

        return np.msort(np.array(result))

    # Error function for deconvolution minimization
    def deconvolution_error(self, args, x_window=None):
        _bands = []

        start = 0
        end = self.x.size

        if x_window is not None:
            start = int(self.find_closest(x_window[0]))
            end = int(self.find_closest(x_window[1]))

  
       

        for i in range(0, len(args), 3):
            _bands.append(Data(self.x, Lorentzian(args[i], args[i + 1], args[i + 2]).eval(self.x)))

        res = np.sum(
            np.square(
                (self.y[start:end] - np.sum(_bands).y[start:end])/self.stds[start:end]
                )
            )
        return res
    
   


    # Deconvolutes a data set
    def deconvolute(self, units: str,
                    x_window: tuple = None,
                    slope_threshold: float = -0.00001,
                    amp_threshold: float = 0.1,
                    _window: int = None,
                    _conv_threshold: int = None,
                    centers_given=None,
                    heights_given=None,
                    save_name: str = None):

        conv_threshold = 3

        if conv_threshold is not None:
            conv_threshold = _conv_threshold

        window = 10

        if window is not None:
            window = _window

        if centers_given is None:
            _centers = list(self.find_peaks_and_shoulders(x_window=x_window,
                                                          slope_threshold=slope_threshold,
                                                          amp_threshold=amp_threshold,
                                                          _window=_window,
                                                          _conv_threshold=_conv_threshold))
        else:
            _centers = centers_given

        _centers = list(set(_centers))
        inputs = []
        i = 0
        for center in _centers:
            if heights_given is None:
                inputs += [self.y[self.find_closest(center)], center, 5]
            else:
                inputs += [heights_given[i], center, 5]
            i += 1

        if save_name is not None:
            if not file_empty(save_name):
                result_dict = open(save_name, "rb")
                oldDict = pickle.load(result_dict)
                result_dict.close()
                if tuple(inputs) in oldDict:
                    return oldDict[tuple(inputs)]
            else:
                result_dict = open(save_name, "wb")
                pickle.dump({" ": " "}, result_dict)
                result_dict.close()

        bounds = []
        for i in range(0, len(inputs), 3):
            #bounds += [(0, abs(inputs[i])*10), (inputs[i + 1] - 10, inputs[i + 1] + 10), (None, None)]
            bounds += [(None, None), (None, None), (None, None)]
        
        pbar = tqdm(total=np.inf,
                    desc=F"\"{self.data_label}\": Deconvoluting {len(_centers)} bands",
                    unit=" iterations")

        def pbar_update(inputs):
            pbar.update(1)

        result = minimize(self.deconvolution_error, inputs,
                          args=(x_window,),
                          method="CG",
                          #bounds=bounds,
                          callback=pbar_update,
                          tol=1E-10,
                          options={'maxiter': 1000000})

        pbar.close()

        _res = []

        print(result.message)
        print(F"Error: {np.round(result.fun, decimals=5)}")
        print("Bands:")
        print("# \tc    \th    \tw   \tFWHM")
        
        _resGauss = []

        for i in range(0, len(list(result.x)), 3):
            _res.append(Data(self.x, Lorentzian(list(result.x)[i],
                                              list(result.x)[i + 1],
                                              list(result.x)[i + 2]).eval(self.x),
                             data_label=F"Band {i // 3}"))
            _resGauss.append(Lorentzian(list(result.x)[i],
                                              list(result.x)[i + 1],
                                              list(result.x)[i + 2]))
            if result.success:
                print(F"{i // 3}\t"
                      F"{np.round((result.x)[i + 1], decimals=5)}\t"
                      F"{np.round(list(result.x)[i], decimals=5)}\t"
                      F"{np.round(list(result.x)[i + 2], decimals=5)}\t"
                      F"{np.round(2 * np.sqrt(2 * np.log(2)) * list(result.x)[i + 2], decimals=5)}")

        band_sum = np.sum(_res)

        band_sum.data_label = "Sum"

        if save_name is not None:
            if not file_empty(save_name):
                result_dict = open(save_name, "rb")
                oldDict = pickle.load(result_dict)
                result_dict.close()
                oldDict[tuple(inputs)] = (_res, [band_sum], result)
                result_dict = open(save_name, "wb")
                pickle.dump(oldDict, result_dict)
                result_dict.close()
        
        res_obj = Fit()

        res_obj.subfunc_data = _res
        res_obj.fit_data = band_sum
        res_obj.result = result
        res_obj.subfuncs = sorted(_resGauss, key=lambda x: x.c)

        return res_obj

    # Error function for deconvolution minimization
    def exgauss_deconvolution_error(self, args, x_window=None):
        _bands = []

        start = 0
        end = self.x.size


        if x_window is not None:
            start = int(self.find_closest(x_window[0]))
            end = int(self.find_closest(x_window[1]))

        size = np.abs(end - start)

        for i in range(0, len(args), 4):
            _bands.append(Data(self.x, ExGaussian(args[i], args[i + 1], args[i + 2], args[i + 3]).eval(self.x)))
        res = np.sum(
            np.square(
                (self.y[start:end] - np.sum(_bands).y[start:end])/self.stds[start:end]
                )
            )

        
        return res
    
   


    # Deconvolutes a data set
    def exgauss_deconvolute(self, units: str,
                    x_window: tuple = None,
                    slope_threshold: float = -0.00001,
                    amp_threshold: float = 0.1,
                    _window: int = None,
                    _conv_threshold: int = None,
                    centers_given=None,
                    heights_given=None,
                    save_name: str = None):

        conv_threshold = 3

        if conv_threshold is not None:
            conv_threshold = _conv_threshold

        window = 10

        if window is not None:
            window = _window

        if centers_given is None:
            _centers = list(self.find_peaks_and_shoulders(x_window=x_window,
                                                          slope_threshold=slope_threshold,
                                                          amp_threshold=amp_threshold,
                                                          _window=_window,
                                                          _conv_threshold=_conv_threshold))
        else:
            _centers = centers_given

        _centers = list(set(_centers))
        inputs = []
        i = 0
        for center in _centers:
            if heights_given is None:
                inputs += [self.y[self.find_closest(center)], center, 1, 1]
            else:
                inputs += [heights_given[i], center, 1, 1]
            i += 1

        if save_name is not None:
            if not file_empty(save_name):
                result_dict = open(save_name, "rb")
                oldDict = pickle.load(result_dict)
                result_dict.close()
                if tuple(inputs) in oldDict:
                    return oldDict[tuple(inputs)]
            else:
                result_dict = open(save_name, "wb")
                pickle.dump({" ": " "}, result_dict)
                result_dict.close()

        bounds = []
        for i in range(0, len(inputs), 4):
            bounds += [(abs(inputs[i]) / 2, abs(inputs[i]) * 4), (inputs[i + 1] - 8, inputs[i + 1] + 8), (0.001, 100), (0.01, 100)]

        pbar = tqdm(total=np.inf,
                    desc=F"\"{self.data_label}\": Deconvoluting {len(_centers)} bands",
                    unit=" iter")

        def pbar_update(inputs):
            pbar.update(1)

        result = minimize(self.exgauss_deconvolution_error, inputs,
                          args=(x_window,),
                          method="SLSQP",
                          bounds=bounds,
                          callback=pbar_update,
                          tol=1E-10,
                          options={'maxiter': 1000000})

        pbar.close()

        _res = []

        print(result.message)
        print(F"Error: {np.round(result.fun, decimals=5)}")
        print("Bands:")
        print("# \tc    \th    \tw   \tl")
        
        _resGauss = []

        for i in range(0, len(list(result.x)), 4):
            _res.append(Data(self.x, ExGaussian(
                list(result.x)[i],
                list(result.x)[i+1],
                list(result.x)[i+2], 
                list(result.x)[i+3]).eval(self.x),
                             data_label= " "))
            _resGauss.append(ExGaussian(list(result.x)[i],
                                        list(result.x)[i + 1],
                                        list(result.x)[i + 2], 
                                        list(result.x)[i + 3]))
            if result.success:
                print(F"{i // 3}\t"
                      F"{np.round((result.x)[i + 1], decimals=5)}\t"
                      F"{np.round(list(result.x)[i], decimals=5)}\t"
                      F"{np.round(list(result.x)[i + 2], decimals=5)}\t"
                      F"{np.round(list(result.x)[i + 3], decimals=5)}")

        band_sum = np.sum(_res)

        band_sum.data_label = F"{self.data_label} Fit"

        
        
        res_obj = Fit()

        _res.sort(key = lambda d: d.x[d.find_closest(np.max(d.y), y=True, x=False)])

        for i in range(len(_res)):
            _res[i].data_label = F"Distr. {i + 1}"

        res_obj.subfunc_data = _res
        res_obj.fit_data = band_sum
        res_obj.result = result
        res_obj.residual_data = Data(self.x, self.y - res_obj.fit_data.y)
        res_obj.subfuncs = sorted(_resGauss, key=lambda x: x.c)

        if save_name is not None:
            if not file_empty(save_name):
                result_dict = open(save_name, "rb")
                oldDict = pickle.load(result_dict)
                result_dict.close()
                oldDict[tuple(inputs)] = res_obj
                result_dict = open(save_name, "wb")
                pickle.dump(oldDict, result_dict)
                result_dict.close()

        return res_obj

    #inputs is list of tuples of T and B
    def trifit_error(self, args, x_window=None):
        start = 0
        end = self.x.size

        if x_window is not None:
            start = int(self.find_closest(x_window[0]))
            end = int(self.find_closest(x_window[1]))

        _fit_vals = np.zeros(np.size(self.x[start:end]))

        for i in range(0, len(args), 2):
            T, B = args[i], args[i+1]
            _fit_vals += ExponentialDecay(T, B).eval(self.x[start:end])

        return sum((self.y[start:end] - _fit_vals)**2)
    
    def tri_exponential_fit(self, inputs, 
                            x_window=None):
        
        start = 0
        end = self.x.size

        if x_window is not None:
            start = int(self.find_closest(x_window[0]))
            end = int(self.find_closest(x_window[1]))

        

        
        
        
        result = minimize(self.trifit_error, inputs,
                          args=(x_window,),
                          method="Powell",
                          tol=1E-15,
                          options={'maxiter': 100000000})



        print(F'''
        Error: {result.fun}
        Tau (ns): {result.x[0]*1E9}
        B: {result.x[1]}
        ''')
        
        fit_vals = result.x
        fit_func = ExponentialDecay(fit_vals[0], fit_vals[1])
        fit_data = Data(self.x[start:end], fit_func.eval(self.x[start:end]))
        return {"data": fit_data,
                "Tau": result.x[0]*1E9}

    def linear_regression(self, x_window=None, verbose=False):

        start = 0
        end = self.x.size

        if x_window is not None:
            print(self.find_closest(x_window[0]))
            start = int(self.find_closest(x_window[0]))
            end = int(self.find_closest(x_window[1]))

        def line(x, m, b):
            return m * x + b

        result = curve_fit(line, self.x[start:end], self.y[start:end])

        m, b = result[0]

        residuals = self.y[start:end] - Line(m, b).eval(self.x[start:end])
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((self.y[start:end] - np.mean(self.y[start:end])) ** 2)
        r_sqrd = 1 - (ss_res / ss_tot)
        adj_r_sqrd = 1 - ((((1 - r_sqrd)**2) * (self.y.size - 1))/(self.y.size - 1 - 1))
        if verbose:
            print(F"{self.data_label}")
            print(F"X int: {-b / m}")
            print(F"m:{m}, b:{b}")
            print(F"adj R^2: {adj_r_sqrd}")
            print(F"R^2: {r_sqrd}")

        return {"data": Data(self.x[start:end], Line(m, b).eval(self.x[start:end]),
                             data_label=self.data_label),
                "slope": m,
                "intercept": b,
                "adj R^2": adj_r_sqrd,
                "R^2": r_sqrd, 
                "func": Line(m, b)}





def sim_Mass_spec(file_name):
    data = open(file_name, "r")
    x = []
    y = []
    for line in data.read().splitlines():
        x.append(float(line.split(",")[0]))
        y.append(float(line.split(",")[1]))
    return Data(x, y, file_name=file_name)


def read_dat(file_name):
    file = open(file_name, "rb")
    for byte in file.read().splitlines():
        print(byte)
###Depreciated
# def pretty_plot(listOfData, xaxis_label, yaxis_label, listOfLabels, xlim=(300, 800), ylim=None, xsize=15, ysize=10,
#                 listOfColors=["black"], figNum=1, title=""):
#     plt.figure(figsize = (xsize, ysize))
#     for dataset, label, color in zip(listOfData, listOfLabels, listOfColors):
#         dataset.plot(figNum, label = label, color = color)
#         plt.xlabel(xaxis_label, fontsize = 30)
#         plt.ylabel(yaxis_label, fontsize = 30)
#         plt.title(title)
#         plt.legend(loc = "best", fontsize = 24)
#         plt.yticks([])
#         plt.xticks(fontsize = 24)
#     plt.xlim(xlim)
#     if ylim != None:
#         plt.ylim(ylim)
#     plt.savefig(title + ".png", bbox_inches = 'tight')
