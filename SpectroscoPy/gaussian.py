import numpy as np
import scipy.special as sp

def helper(self, x):
    z = (1 / np.sqrt(2)) * ((self.w / self.l) - ((x - self.c)/self.w))

    if z<0:
        #Equation 1
        A1 = ((self.h * self.w)/self.l) * np.sqrt(np.pi / 2)

        EXP1 = np.exp(0.5 * np.square(self.w /self.l) - ((x - self.c) / self.l))

        ERFC1 = sp.erfc((1 / np.sqrt(2)) * ((self.w / self.l) - ((x - self.c) / self.w)))

        return A1 * EXP1 * ERFC1

    elif (0 <= z) and (z <= 6.71E7):
        #Equation 2
        A2 = self.h * (self.w/self.l) * (np.sqrt(np.pi / 2))

        EXP2 = np.exp(-0.5 * np.square((x - self.c)/self.w))

        ERFCX2 = sp.erfcx((1/np.sqrt(2)) * ((self.w/self.l) - ((x - self.c)/self.w)))

        return A2 * EXP2 * ERFCX2

    else:
        #Equation 3
        A3 = self.h

        EXP3 = np.exp(-0.5 * np.square((x - self.c)/self.w))

        DENOM3 = 1 + (self.l * (x - self.c))/(np.square(self.w))

        return (A3 * EXP3)/DENOM3

helper = np.vectorize(helper)


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
        self.FWHM = 2 * np.sqrt(2 * np.log(2)) * self.w

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
    
    def area(self):
        return np.sqrt(2) * self.h * np.abs(self.w) * np.sqrt(np.pi)


# Gaussian function with height h, center c, and width (sigma) w
class ExGaussian(object):

    def __init__(self, h: float, c: float, w: float, l: float):
        """
        Initializes a new gaussian object with specified parameters.

        :param h: maximum value on the y-axis for the function
        :param c: position of the maximum value on the x axis
        :param w: spread of the curve (larger numbers -> more spread out)
        """
        self.h = h
        self.c = c
        self.w = w
        self.l = l
    

    def exgaussian(self, x: float) -> float:
        """
        Function that holds the mathematical definition of the specified gaussian function.
        Generally not called directly. To evaluate the gaussian at some x value or array of x values,
        use eval(x) instead.
        """
        
        if isinstance(x, np.ndarray):
            return helper(self, x)
        else:
            return helper(self, x)

        

    def eval(self, x: any) -> any:
        """
        Evaluates the gaussian function at some float x or for every float in a 1D numpy array. If the input is an
        array, then the output has the same dimensions.

        :param x: position or array of positions to evaluate the gaussian function at
        :return: y-value or array of y-values on the gaussian function after evaluation.
        """
        return self.exgaussian(x)