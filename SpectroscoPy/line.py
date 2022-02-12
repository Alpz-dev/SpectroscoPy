import numpy as np

class Line(object):
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def line(self, x):
        return self.m*x + self.b

    def eval(self, x):
        return self.line(x)