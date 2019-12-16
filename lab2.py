import numpy as np
from lab1 import BaseSchema, AlternatingDirectionMethod
from math import sin, cos
from functools import partial


def _precise_solution(x, y):
    return 4 * (2 * x - 3 * y) * sin(4 * x + 3 * y) + 4 * y ** 2


def _poison_heterogenity(x, y):
    return 8 - 8 * cos(4 * x + 3 * y) - 100 * (2 * x - 3 * y) * sin(4 * x + 3 * y)


def solve_equation_min_risidual(equation_matrix, result_function, epsilon):
    """
    Решение уравнения методом минимальных невязок
    :param equation_matrix: матрица коэффицентов СЛАУ
    :param result_function: Правая часть СЛАУ
    :param epsilon: Критерий останова
    :return:
    """
    def stop_criterion(_risidual, _result_function):
        actual = np.dot(_risidual, _risidual) / np.dot(_result_function, _result_function)
        return actual >= epsilon
    u = np.random.rand(*result_function.shape)
    criterion = True
    while criterion:
        risidual = (equation_matrix @ u - result_function)
        a_r = (equation_matrix @ risidual)
        tau = np.dot(a_r, risidual) / np.dot(a_r, a_r)
        u_next = u - tau * risidual
        criterion = stop_criterion(risidual, result_function)
        u = u_next
    return u


if __name__ == "__main__":
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
    solver.matrix_solver = partial(solve_equation_min_risidual, epsilon=0.0000001)
    solver.run()
    solver.plot_error()
