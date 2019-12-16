import numpy as np
from lab1 import get_count
from plotter import plot_rids, plt


def precise(x):
    return x ** 5 / 20 - 1 / 20 * x


def heterogenity(x):
    return x ** 3


class FiniteElementMethod:
    def __init__(self, x_left, x_right, step):
        """
        Метод конечных элементов(Ритца), граничные условия нулевые
        :param x_left: левая граница по х
        :param x_right: правая граница
        :param step:
        """
        self._x_left = x_left
        self._x_right = x_right
        self._step = step
        self._count = get_count(x_left, x_right, step)
        self._matrix = np.zeros((self._count - 2, self._count - 2))
        self._right_part_eq = np.zeros((self._count - 2,))
        self._x_net = [x_left + i * step for i in range(self._count)]
        self._matrix.fill(1 / step)
        np.fill_diagonal(self._matrix, -2 / self._step)
        self._matrix *= np.tri(*self._matrix.shape, k=1)
        self._matrix *= 1 - np.tri(*self._matrix.shape, k=-2)
        self._precise = np.array([precise(self._x_net[i]) for i in range(1, self._count - 1)])
        self._c = np.zeros((self._count - 2,))
        self.fill_right_eq_part()
        self._u = np.zeros(self._count - 2)
        self._numeric_mattr = np.eye(self._count - 2)
        self._numeric_mattr.fill(1 / step ** 2)
        np.fill_diagonal(self._numeric_mattr, -2 / self._step ** 2)
        self._numeric_mattr *= np.tri(*self._numeric_mattr.shape, k=1)
        self._numeric_mattr *= 1 - np.tri(*self._numeric_mattr.shape, k=-2)
        self._f = [heterogenity(self._x_net[i]) for i in range(1, self._count - 1)]
        self._num_sol = [0] * (self._count - 2)

    def fill_right_eq_part(self):
        """
        Заполнение правой части СЛАУ
        :return:
        """
        for i in range(1, self._count - 1):
            first = self._x_net[i] ** 5 * 2 / 5 + self._x_net[i + 1] ** 5 / 20
            second = - self._x_net[i] ** 4 * (self._x_net[i + 1] + self._x_net[i - 1]) / 4
            third = self._x_net[i - 1] ** 5 / 20
            self._right_part_eq[i - 1] = 1 / self._step * (first + second + third)

    def phi_i(self, x, i):
        if 0 <= x < self._x_net[i - 1]:
            return 0
        elif self._x_net[i - 1] <= x < self._x_net[i]:
            return 1 / self._step * (x - self._x_net[i - 1])
        elif self._x_net[i] <= x < self._x_net[i + 1]:
            return -1 / self._step * (x - self._x_net[i + 1])
        else:
            return 0

    def solve(self):
        self._c = np.linalg.solve(self._matrix, self._right_part_eq)

    def get_max_err(self):
        return np.amax(self._c - self._precise)

    def get_num_sol(self):
        self._num_sol = np.linalg.solve(self._numeric_mattr, self._f)

    def get_errors(self):
        return np.abs(self._c - self._precise)

    def plot(self):
        plot_rids(self._x_net[1:-1], self._c, self._precise, self._num_sol)

# method = FiniteElementMethod(
#     x_left=0,
#     x_right=1,
#     step=0.2
# )
# method.get_num_sol()
# method.solve()
# method.plot()


err_net = [0.2, 0.1, 0.05, 0.025, 0.01, 0.005]
methods = [
    FiniteElementMethod(
        x_left=0,
        x_right=1,
        step=0.01
    )
    for i in err_net
]
for method in methods:
    method.solve()

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(err_net, [method.get_max_err() for method in methods], '-s')
plt.grid(True)
plt.show()