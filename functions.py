import numpy as np


class Function:

    def __init__(self, xmin, xmax, ymin, ymax, step, minima=None, initial=None):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.step = step

        self.minima = minima
        self.initial = initial

    def __call__(self, x, y):
        raise NotImplementedError

    def grad(self, x, y):
        raise NotImplementedError


class Beale(Function):

    def __init__(self, xmin, xmax, ymin, ymax, step, initial=None):
        super().__init__(xmin=xmin, xmax=xmax,
                         ymin=ymin, ymax=ymax,
                         step=step, minima=np.array([[3.0, 0.5]]),
                         initial=initial)

    def __call__(self, x, y):
        return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2

    def grad(self, x, y):
        dx = -12.75 + 3 * y + 4.5 * (y ** 2) + 5.25 * (y ** 3) + 2 * x * (
                3 - 2 * y - (y ** 2) - 2 * (y ** 3) + (y ** 4) + (y ** 6))
        dy = 6 * x * (0.5 + 1.5 * y + 2.625 * (y ** 2) + x * (
                -0.333333 - 0.333333 * y - (y ** 2) + 0.666667 * (y ** 3) + (y ** 5)))
        return np.array([dx, dy])


class Booth(Function):

    def __init__(self, xmin, xmax, ymin, ymax, step, initial=None):
        super().__init__(xmin=xmin, xmax=xmax,
                         ymin=ymin, ymax=ymax,
                         step=step, minima=np.array([[1.0, 3.0]]),
                         initial=initial)

    def __call__(self, x, y):
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

    def grad(self, x, y):
        dx = 10 * x + 8 * y - 34
        dy = 8 * x + 10 * y - 38
        return np.array([dx, dy])


class Himmelblau(Function):

    def __init__(self, xmin, xmax, ymin, ymax, step, initial=None):
        super().__init__(xmin=xmin, xmax=xmax,
                         ymin=ymin, ymax=ymax,
                         step=step, minima=np.array([[3.0, 2.0],
                                                     [-2.805118, 3.131312],
                                                     [-3.779310, -3.283186],
                                                     [3.584428, -1.848126]]),
                         initial=initial)

    def __call__(self, x, y):
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    def grad(self, x, y):
        dx = 2 * (-7 + x + (y ** 2) + 2 * x * (-11 + (x ** 2) + y))
        dy = 2 * (-11 + (x ** 2) + y + 2 * y * (-7 + x + (y ** 2)))
        return np.array([dx, dy])


class CamelBack(Function):

    def __init__(self, xmin, xmax, ymin, ymax, step, initial=None):
        super().__init__(xmin=xmin, xmax=xmax,
                         ymin=ymin, ymax=ymax,
                         step=step, minima=np.array([[0.0898, -0.7126],
                                                     [-0.0898, 0.7126]]),
                         initial=initial)

    def __call__(self, x, y):
        return (4 - (2.1 * x ** 2) + ((x ** 4) / 3)) * x ** 2 + x * y + (-4 + 4 * y ** 2) * y ** 2

    def grad(self, x, y):
        raise NotImplementedError


class Saddle(Function):

    def __init__(self, xmin, xmax, ymin, ymax, step, initial=None):
        super().__init__(xmin=xmin, xmax=xmax,
                         ymin=ymin, ymax=ymax,
                         step=step, minima=np.array([[0.0, 0.0]]),
                         initial=initial)

    def __call__(self, x, y):
        return x * x - y * y

    def grad(self, x, y):
        dx = 2 * x
        dy = -2 * y
        return np.array([dx, dy])
