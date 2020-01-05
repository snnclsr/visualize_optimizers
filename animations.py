#################################################################################################
# Code is taken from:                                                                           #
# http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/  #
#                                                                                               #
#################################################################################################

import matplotlib
import matplotlib.pyplot as plt

from matplotlib import animation
from itertools import zip_longest


class TrajectoryAnimation(animation.FuncAnimation):

    def __init__(self, *paths, labels=[], fig=None, ax=None, frames=None,
                 interval=60, repeat_delay=5, blit=True, **kwargs):

        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
        else:
            if ax is None:
                ax = fig.gca()

        self.fig = fig
        self.ax = ax

        self.paths = paths

        if frames is None:
            frames = max(path.shape[1] for path in paths)

        self.lines = [ax.plot([], [], label=label, lw=2)[0]
                      for _, label in zip_longest(paths, labels)]
        self.points = [ax.plot([], [], 'o', color=line.get_color())[0]
                       for line in self.lines]

        super(TrajectoryAnimation, self).__init__(fig, self.animate, init_func=self.init_anim,
                                                  frames=frames, interval=interval, blit=blit,
                                                  repeat_delay=repeat_delay, **kwargs)

    def init_anim(self):
        for line, point in zip(self.lines, self.points):
            line.set_data([], [])
            point.set_data([], [])
        return self.lines + self.points

    def animate(self, i):
        for line, point, path in zip(self.lines, self.points, self.paths):
            line.set_data(*path[::, :i])
            point.set_data(*path[::, i - 1:i])
        return self.lines + self.points