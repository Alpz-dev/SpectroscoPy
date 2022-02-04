import numpy as np

class Lorentzian(object):
    def __init__(self, h, c, w):
        self.h  = h
        self.c = c
        self.w = w

    def lorentzian(self, x):
        return ((1/np.pi)*(0.5*self.w))/((x - self.c)**2 + (0.5*self.w)**2)

    def eval(self, x):
        return self.lorentzian(x)

# Object-Oriented Gaussian Function
class Gaussian(object):
    def __init__(self, h, c, w):
        self.h = h
        self.c = c
        self.w = w

    def gaussian(self, x):
        return self.h * np.exp(-np.square(x - self.c) / (2 * np.square(self.w)))

    def eval(self, x: any) -> any:
        return self.gaussian(x)