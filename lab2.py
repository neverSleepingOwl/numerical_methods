import numpy as np
from lab1 import BaseSchema, AlternatingDirectionMethod
from math import sin, cos
from functools import partial


def _precise_solution(x, y):
    # return 5 * (-3 * x + 4 * y) * cos(5 * x + 2 * y) + 3 * y ** 2
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
        #
        actual = np.dot(_risidual, _risidual) / np.dot(_result_function, _result_function)
        # print("==============")
        # print(actual)
        # print(epsilon)
        return actual >= epsilon
    u = np.random.rand(*result_function.shape)
    # print(u)
    criterion = True
    while criterion:
        risidual = (equation_matrix @ u - result_function)
        a_r = (equation_matrix @ risidual)
        tau = np.dot(a_r, risidual) / np.dot(a_r, a_r)
        u_next = u - tau * risidual
        criterion = stop_criterion(risidual, result_function)
        u = u_next
    return u


# class MinRisidualPoissonCrossSchema(BaseSchema):
#     def generate_matrix(self):
#         self._matr_dim = self._x_count * self._y_count
#         self._matrix = np.zeros((self._x_count * self._y_count, self._x_count * self._y_count))
#         self._func = np.zeros((self._matr_dim))
#         count = 0
#         for i in range(1, self._x_count - 1):
#             for j in range(1, self._y_count - 1):
#                 self._matrix[count][j * self._x_count + i] = -2 * (1 / self._x_step ** 2 + 1 / self._y_step ** 2)
#                 self._matrix[count][j * self._x_count + i - 1] = 1 / self._x_step ** 2
#                 self._matrix[count][j * self._x_count + i + 1] = 1 / self._x_step ** 2
#                 self._matrix[count][(j + 1) * self._x_count + i] = 1 / self._y_step ** 2
#                 self._matrix[count][(j - 1) * self._x_count + i] = 1 / self._y_step ** 2
#                 self._func[count] = self._heterogenity(
#                     self._x_net[i],
#                     self._y_net[j]
#                 )
#                 count += 1
#         for i in range(1, self._y_count - 1):
#             self._matrix[count][i * self._x_count] = 1
#             self._func[count] = self._precise(x=self._x_net[0], y=self._y_net[i])
#             count += 1
#             self._matrix[count][i * self._x_count + self._x_count - 1] = 1
#             self._func[count] = self._precise(x=self._x_net[-1], y=self._y_net[i])
#             count += 1
#         for j in range(self._x_count):
#             self._matrix[count][j] = 1
#             self._func[count] = self._precise(x=self._x_net[j], y=self._y_net[0])
#             count += 1
#             self._matrix[count][(self._y_count - 1) * self._x_count + j] = 1
#             self._func[count] = self._precise(x=self._x_net[j], y=self._y_net[-1])
#             count += 1
#
#     def iteration(self):
#         solution_vec = solve_equation_min_risidual(self._matrix, self._func.transpose(), 0.0001)
#         print(np.linalg.cond(self._matrix))
#         # solution_vec = np.linalg.solve(self._matrix, self._func)
#         self._err = 0.00001
#         for i in range(self._x_count):
#             for j in range(self._y_count):
#                 self._u[j][i] = solution_vec[j * self._x_count + i]


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
