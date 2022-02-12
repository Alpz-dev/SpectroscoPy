import numpy as np

class ExponentialDecay(object):
    def __init__(self, T, B):
        self.T = T
        self.B = B


    def exponential_decay(self, x):
        return (self.B*np.exp(-x/self.T))

    def eval(self, x):
        return self.exponential_decay(x)
