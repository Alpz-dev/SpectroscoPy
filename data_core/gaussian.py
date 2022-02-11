import numpy as np


# Lorentzian function with height h, center c, and width (sigma) w
class Lorentzian(object):
    def __init__(self, h: float, c: float, w: float):
        """
        Initializes a new lorentzian object with specified parameters.

        :param h: maximum value on the y-axis for the function
        :param c: position of the maximum value on the x axis
        :param w: spread of the curve (larger numbers -> more spread out)
        """
        self.h = h
        self.c = c
        self.w = w

    def lorentzian(self, x: float) -> float:
        """
        Function that holds the mathematical definition of the specified lorentzian function.
        Generally not called directly. To evaluate the lorentzian at some x value or array of x values,
        use eval(x) instead.
        """
        return ((1 / np.pi) * (0.5 * self.w)) / ((x - self.c) ** 2 + (0.5 * self.w) ** 2)

    def eval(self, x: any) -> any:
        """
        Evaluates the lorentzian function at some float x or for every float in a 1D numpy array. If the input is an
        array, then the output has the same dimensions.

        :param x: position or array of positions to evaluate the lorentzian function at
        :return: y-value or array of y-values on the lorentzian function after evaluation.
        """
        return self.lorentzian(x)


# Gaussian function with height h, center c, and width (sigma) w
class Gaussian(object):

    def __init__(self, h: float, c: float, w: float):
        """
        Initializes a new gaussian object with specified parameters.

        :param h: maximum value on the y-axis for the function
        :param c: position of the maximum value on the x axis
        :param w: spread of the curve (larger numbers -> more spread out)
        """
        self.h = h
        self.c = c
        self.w = w

    def gaussian(self, x: float) -> float:
        """
        Function that holds the mathematical definition of the specified gaussian function.
        Generally not called directly. To evaluate the gaussian at some x value or array of x values,
        use eval(x) instead.
        """
        return self.h * np.exp(-np.square(x - self.c) / (2 * np.square(self.w)))

    def eval(self, x: any) -> any:
        """
        Evaluates the gaussian function at some float x or for every float in a 1D numpy array. If the input is an
        array, then the output has the same dimensions.

        :param x: position or array of positions to evaluate the gaussian function at
        :return: y-value or array of y-values on the gaussian function after evaluation.
        """
        return self.gaussian(x)
