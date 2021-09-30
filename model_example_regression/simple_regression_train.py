import numpy as np
import math
import pickle


def fit(X, Y):
    """
    From https://code.activestate.com/recipes/578914-simple-linear-regression-with-pure-python/
    """

    def mean(Xs):
        return sum(Xs) / len(Xs)

    m_X = mean(X)
    m_Y = mean(Y)

    def std(Xs, m):
        normalizer = len(Xs) - 1
        return math.sqrt(sum((pow(x - m, 2) for x in Xs)) / normalizer)

    def pearson_r(Xs, Ys):

        sum_xy = 0
        sum_sq_v_x = 0
        sum_sq_v_y = 0

        for (x, y) in zip(Xs, Ys):
            var_x = x - m_X
            var_y = y - m_Y
            sum_xy += var_x * var_y
            sum_sq_v_x += pow(var_x, 2)
            sum_sq_v_y += pow(var_y, 2)
        return sum_xy / math.sqrt(sum_sq_v_x * sum_sq_v_y)

    r = pearson_r(X, Y)

    b = r * (std(Y, m_Y) / std(X, m_X))
    A = m_Y - b * m_X

    def line(x):
        return b * x + A

    return line, [b, A]


if __name__ == "__main__":

    X = np.array([1, 2, 3, 5, 22, -10])
    Y = 2.5*X + 3  # y = 1 * x_0 + 2 * x_1 + 3

    model, model_params = fit(X, Y)
    print('2', model(2))
    print('-1', model(-1))
    print('0', model(0))

    pickle.dump(model_params, open(
        '../data/models/simple_regression.pkl', 'wb'))
