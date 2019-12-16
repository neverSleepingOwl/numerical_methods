from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from typing import List, Union, Callable
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def plot_3d_function(
        x_array: List[float],
        y_array: List[float],
        u: Union[np.array, Callable[[float, float], float]]
):
    if callable(u):
        arr = np.zeros((len(x_array), len(y_array)))
        for i in range(len(y_array)):
            for j in range(len(x_array)):
                arr[i][j] = u(x_array[j], y_array[i])
    else:
        arr = u
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    _x, _y = np.meshgrid(x_array, y_array)
    surf = ax.plot_surface(
        _x,
        _y,
        np.array(arr),
        cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-20, 20)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(30, 0)
    plt.draw()
    plt.pause(10)


def plot_rids(x_net, y_net, precise=None, diff=None):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x_net, y_net, '-s')
    if precise is not None:
        plt.plot(x_net, precise, '-s')
    if diff is not None:
        plt.plot(x_net, diff, '-s')
    plt.legend(["Ридс", "точное", "Конечно-разностный"])
    plt.grid(True)
    plt.show()
