import numpy as np


class Function:

    def __init__(self, xmin, xmax, ymin, ymax, step, minima=None):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.step = step

        self.minima = minima

    def __call__(self, x, y):
        pass

class Beale(Function):

    def __init__(self, xmin, xmax, ymin, ymax, step=0.2):
        super().__init__(xmin=xmin, xmax=xmax,
                         ymin=ymin, ymax=ymax,
                         step=step, minima=np.array([3.0, 0.5]))

    def __call__(self, x, y):
        return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2


class Booth(Function):

    def __init__(self, xmin, xmax, ymin, ymax, step):
        super().__init__(xmin=xmin, xmax=xmax,
                         ymin=ymin, ymax=ymax,
                         step=step, minima=np.array([1.0, 3.0]))

    def __call__(self, x, y):
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
