"""
Решения задачи Дирихле уравнения Пуассона
с неоднородными граничными условиями,
с помощью метода установления
явной разностной схемой и схемой метода
переменных направлений (вариант 3)
"""
from math import cos, sin, pi, ceil
from typing import Callable
import numpy as np
from plotter import plot_3d_function
# from tdma import prog
import sympy
# import warnings
# warnings.filterwarnings('error')


def _poison_heterogenity(x: float, y: float) -> float:
    return 65 * (2 * x - 3 * y) * cos(3 * x + 2 * y) + 6


def _precise_solution(x: float, y: float) -> float:
    return 5 * (3 * y - 2 * x) * cos(3 * x + 2 * y) + 3 * y ** 2


# def tdma(a, b, c, f):
#     a, b, c, f = list(map(lambda k_list: list(map(float, k_list)), (a, b, c, f)))
#
#     alpha = [0]
#     beta = [0]
#     n = len(f)
#     x = [0] * n
#
#     for i in range(n - 1):
#         alpha.append(-b[i] / (a[i] * alpha[i] + c[i]))
#         beta.append((f[i] - a[i] * beta[i]) / (a[i] * alpha[i] + c[i]))
#
#     x[n - 1] = (f[n - 1] - a[n - 2] * beta[n - 1]) / (c[n - 1] + a[n - 2] * alpha[n - 1])
#
#     for i in reversed(range(n - 1)):
#         x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]
#
#     return x


def get_count(left_border: float, right_border: float, step: float) -> int:
    """
    Ищем число шагов по x и по y
    :param left_border: левая граница
    :param right_border: правая граница
    :param step:  шаг
    :return: число шагов
    """
    return ceil((right_border - left_border) / step) + 1


class BaseSchema:
    def __init__(
            self,
            x_left_border: float,
            x_right_border: float,
            y_left_border: float,
            y_right_border: float,
            x_step: float,
            y_step: float,
            precise_solution: Callable[[float, float], float],
            poison_heterogenity: Callable[[float, float], float],
            t_error=0.001
    ):
        """
        Базовый ласс, описывающий решение и Дирихле
        уравнения Пуассона методом установления
        (заполнение начального слоя,
         построение графиков ошибок и данных,
         заполнение граничных условий)

        :param x_left_border: нижняя граница по x
        :param x_right_border: верхняя граница по x
        :param y_left_border: нижняя граница по y
        :param y_right_border: верхняя граница по y
        :param x_step: Шаг по x
        :param y_step: Шаг по y
        :param precise_solution Точное решение для тестов
        :param poison_heterogenity Неоднородность правой части уравнения пуассона
        :param t_error - эпсилон для критерия останова
        """

        self._x_count = get_count(x_left_border, x_right_border, x_step)
        self._y_count = get_count(y_left_border, y_right_border, y_step)
        self._u: np.array = np.zeros((self._y_count, self._x_count))
        self._x_net = [x_left_border + i * x_step for i in range(self._x_count)]
        self._y_net = [y_left_border + i * y_step for i in range(self._y_count)]
        self._x_step = x_step
        self._y_step = y_step
        self._precise = precise_solution
        self._precise_matrix: np.array = np.zeros((self._y_count, self._x_count))
        self._heterogenity = poison_heterogenity
        self._t_step = self._x_step ** 2 * self._y_step ** 2 / (self._x_step ** 2 + self._y_step ** 2) / 2
        self._error_matrix: np.array = np.zeros((self._y_count, self._x_count))
        self._max_diff = 10
        self._err = t_error

    def find_max_error(self) -> float:
        return max(map(max, self._error_matrix))

    def plot_data(self):
        plot_3d_function(
            x_array=self._x_net,
            y_array=self._y_net,
            u=self._u
        )

    def plot_error(self):
        plot_3d_function(
            x_array=self._x_net,
            y_array=self._y_net,
            u=self._error_matrix
        )

    def plot_precise(self):
        plot_3d_function(
            x_array=self._x_net,
            y_array=self._y_net,
            u=self._precise_matrix
        )

    def find_precise_solution(self) -> None:
        for i in range(self._x_count):
            for j in range(self._y_count):
                self._precise_matrix[j][i] = self._precise(
                    self._x_net[i],
                    self._y_net[j]
                )

    def find_errors(self) -> None:
        for i in range(self._x_count):
            for j in range(self._y_count):
                self._error_matrix[j][i] = self._precise_matrix[j][i] - self._u[j][i]

    def iteration(self):
        raise NotImplementedError

    def run(self):
        self.iteration()
        self.find_precise_solution()
        self.find_errors()

    def setup_b_c(self, u):
        for i in range(self._y_count):
            u[i][0] = self._precise(x=self._x_net[0], y=self._y_net[i])
            u[i][1] = self._precise(x=self._x_net[1], y=self._y_net[i])
            u[i][-1] = self._precise(x=self._x_net[-1], y=self._y_net[i])
            u[i][-2] = self._precise(x=self._x_net[-2], y=self._y_net[i])

        for j in range(self._x_count - 1):
            u[0][j] = self._precise(x=self._x_net[j], y=self._y_net[0])
            u[1][j] = self._precise(x=self._x_net[j], y=self._y_net[1])
            u[-1][j] = self._precise(x=self._x_net[j], y=self._y_net[-1])
            u[-2][j] = self._precise(x=self._x_net[j], y=self._y_net[-2])

    def _setup_boundary__conditions(self):
        """
        Заполняем сетку с помощью граничных условий.
        :return: None
        """
        self.setup_b_c(self._u)


class ExplicitCrossSchema(BaseSchema):
    def __init__(self, *args, **kwargs):
        super(ExplicitCrossSchema, self).__init__(*args, **kwargs)
        self._setup_boundary__conditions()

    def __solve_single_iteration(self, iteration_x: int, iteration_y: int):
        """

        :param iteration_x: Индекс по x
        :param iteration_y: Индекс по y
        :return:
        """
        x_derivative = self._u[iteration_y][iteration_x + 1] - \
                       2 * self._u[iteration_y][iteration_x] + \
                       self._u[iteration_y][iteration_x - 1]
        x_derivative /= self._x_step ** 2
        y_derivative = self._u[iteration_y + 1][iteration_x] - \
                       2 * self._u[iteration_y][iteration_x] + \
                       self._u[iteration_y - 1][iteration_x]
        y_derivative /= self._y_step ** 2
        func_value = -1 * self._heterogenity(
            self._x_net[iteration_x],
            self._y_net[iteration_y]
        )
        to_return = self._t_step * (x_derivative + y_derivative + func_value)
        # разность между i + 1 и i слоями по t
        # (По t нам не нужны слои, нужна только разность между текущим слоем и
        # следующим слоем.
        to_assign = to_return + self._u[iteration_y][iteration_x]
        self._u[iteration_y][iteration_x] = to_assign
        return to_return

    def iteration(self) -> None:
        """
        Итерируемся по всей сетке и находим значения явным спусобом
        """
        c = 0
        while self._max_diff >= self._err:
            steps = []
            for j in range(1, self._y_count - 1):
                for i in range(2, self._x_count - 2):
                    steps.append(self.__solve_single_iteration(i, j))

            self._max_diff = max(steps)
            c += 1
            if c % 10 == 0:
                print(c)
                print(self._t_step)
                print(self._max_diff)


def tdma(matrix, f):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    (Tdma or thomas algorithm)
    Three diag matrix algorithm
    '''
    a = np.diag(matrix, k=-1)
    c = np.diag(matrix, k=1)
    b = np.diag(matrix)
    nf = len(f)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, f))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


class AlternatingDirectionMethod(BaseSchema):
    def __init__(self, *args, **kwargs):
        super(AlternatingDirectionMethod, self).__init__(*args, **kwargs)
        self._t_step = (self._x_step * self._y_step) / (2 * sin(pi / (2 * self._x_count)) * cos(pi / (2 * self._x_count)))
        # self._t_step = (self._x_step**2+self._y_step**2)/10
        # self._t_step = 0.002
        self._setup_boundary__conditions()
        self.intermidiate: np.array = np.zeros((self._y_count, self._x_count))
        self.find_precise_solution()

    @property
    def matrix_solver(self):
        return self._matrix_solver

    @matrix_solver.setter
    def matrix_solver(self, val):
        self._matrix_solver = val

    def solve_matrix(self, a, b, c, f):
        matrix = np.zeros((len(a), len(a)))
        for i in range(1, len(a) - 1):
            matrix[i][i] = -b[i]
            matrix[i][i - 1] = a[i]
            matrix[i][i + 1] = c[i]
        matrix[0][0] = -1
        matrix[len(a) - 1][len(a) - 1] = -1
        f = -1 * np.array(f)
        return self.matrix_solver(matrix, f)

    def a_m_y_prog(self):
        return self._t_step / (2 * self._y_step ** 2)

    def c_m_y_prog(self):
        return self._t_step / (self._y_step ** 2) + 1

    def b_m_y_prog(self):
        return self._t_step / (2 * self._y_step ** 2)

    def a_n_x_prog(self):
        return self._t_step / (2 * self._x_step ** 2)

    def c_n_x_prog(self):
        return self._t_step / (self._x_step ** 2) + 1

    def b_n_x_prog(self):
        return self._t_step / (2 * self._x_step ** 2)

    def f_m_y_prog(self, x_index, y_index):
        dt = self._t_step
        hx = self._x_step
        f = self._heterogenity
        u = self._u
        prev_part = dt / (2 * hx ** 2) * (u[y_index][x_index - 1] + u[y_index][x_index + 1])
        t_part = -dt / 2 * f(self._x_net[x_index], self._y_net[y_index])
        final_express = (1 - dt / (hx ** 2)) * u[y_index][x_index]
        return prev_part + t_part + final_express

    def f_n_x_prog(self, x_index, y_index):
        dt = self._t_step
        hy = self._y_step
        f = self._heterogenity
        u = self.intermidiate

        prev_part = dt / (2 * hy ** 2) * (u[y_index - 1][x_index] + u[y_index + 1][x_index])
        t_part = -dt / 2 * f(self._x_net[x_index], self._y_net[y_index])
        final_express = (1 - dt / (hy ** 2)) * u[y_index][x_index]
        return prev_part + t_part + final_express

    def y_prog(self):
        self.setup_b_c(self.intermidiate)
        for i in range(1, self._x_count - 1):
            _a, _b, _c, _f = [0], [0], [1], [self._precise_matrix[0][i]]
            for j in range(1, self._y_count - 1):

                _a.append(self.a_m_y_prog())
                _b.append(self.b_m_y_prog())
                _c.append(self.c_m_y_prog())
                _f.append(self.f_m_y_prog(i, j))
            _a = _a + [0]
            _b = _b + [0]
            _c = _c + [1]
            _f = _f + [self._precise_matrix[-1][i]]

            x = self.solve_matrix(_a, _c, _b, _f)
            for p, val in enumerate(x):
                if 0 < p < len(x) - 1:
                    self.intermidiate[p][i] = val

    def x_prog(self):
        self.setup_b_c(self.intermidiate)
        out: np.array = np.zeros((self._y_count, self._x_count))
        self.setup_b_c(out)
        for i in range(1, self._y_count - 1):
            _a, _b, _c, _f = [0], [0], [1], [self._precise_matrix[i][0]]
            for j in range(1, self._x_count - 1):

                _a.append(self.a_n_x_prog())
                _b.append(self.b_n_x_prog())
                _c.append(self.c_n_x_prog())
                _f.append(self.f_n_x_prog(j, i))
            _a = _a + [0]
            _b = _b + [0]
            _c = _c + [1]
            _f = _f + [self._precise_matrix[i][-1]]
            x = self.solve_matrix(_a, _c, _b, _f)
            for p, val in enumerate(x):
                if 0 < p < len(x) - 1:
                    out[i][p] = val
        return out

    def single_step(self):
        self.y_prog()
        return self.x_prog()

    def iteration(self):
        count = 0
        while self._max_diff >= self._err:
            output = self.single_step()
            x = np.amax(np.array(self._u) - np.array(output))
            if x < self._max_diff:
                self._max_diff = x
            self._u = output
            count += 1
            print(f"Count: {count}")
            print(f"Diff: {x}")


if __name__ == "__main__":
    # solver = ExplicitCrossSchema(
    #     x_left_border=0,
    #     x_right_border=1,
    #     y_left_border=0,
    #     y_right_border=2,
    #     x_step=0.025,
    #     y_step=0.025,
    #     precise_solution=_precise_solution,
    #     poison_heterogenity=_poison_heterogenity
    # )

    solver = AlternatingDirectionMethod(
        x_left_border=0,
        x_right_border=1,
        y_left_border=0,
        y_right_border=2,
        x_step=0.1,
        y_step=0.1,
        precise_solution=_precise_solution,
        poison_heterogenity=_poison_heterogenity
    )
    solver.matrix_solver = tdma
    solver.run()
    print(solver.find_max_error())
    # solver.plot_precise()
    solver.plot_error()
    # solver.plot_precise()
