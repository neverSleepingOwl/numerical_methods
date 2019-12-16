import numpy as np
from lab1 import get_count


def precise(x):
    return x ** 5 / 20


def heterogenity(x):
    return x ** 3


class FiniteElementMethod:
    def __init__(self, x_left, x_right, step):
        """

        :param x_left:
        :param x_right:
        :param step:
        """
        self._x_left = x_left
        self._x_right = x_right
        self._step = step
        self._count = get_count(x_left, x_right, step)
        self._matrix = np.zeros((self._count - 2, self._count - 2))
        self._right_part_eq = np.zeros((self._count - 2,))
        self._x_net = [x_left + i * step for i in range(self._count)]
        self._matrix.fill(-1 / step)
        np.fill_diagonal(self._matrix, 2 / self._step)
        self._matrix *= np.tri(*self._matrix.shape, k=1)
        self._matrix *= 1 - np.tri(*self._matrix.shape, k=-2)
        self._precise = np.array([precise(self._x_net[i]) for i in range(1, self._count - 1)])
        self._u = np.zeros((self._count - 2,))
        self.fill_right_eq_part()

    def fill_right_eq_part(self):
        """
        Заполнение правой части СЛАУ
        :return:
        """
        for i in range(1, self._count - 1):
            first = self._x_net[i] ** 5 * 2 / 5 + self._x_net[i + 1] ** 5 / 20
            second = - self._x_net[i] ** 4 * (self._x_net[i + 1] + self._x_net[i - 1]) / 4
            third = self._x_net[i - 1] ** 5 / 20
            self._right_part_eq[i - 1] = 1 / self._step * first + second + third

    def solve(self):
        self._u = np.linalg.solve(self._matrix.transpose(), self._right_part_eq)
        print(self._matrix @ self._precise)
        print(self._right_part_eq)
        print(len(self._precise))
        print(len(self._right_part_eq))
        print(self._matrix.shape)
    def get_max_err(self):
        return np.amax(self._u - self._precise)


method = FiniteElementMethod(
    x_left=0,
    x_right=1,
    step=0.2
)

method.solve()
print(method.get_max_err())
