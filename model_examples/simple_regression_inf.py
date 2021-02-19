import numpy as np
import pickle
import matplotlib.pyplot as plt


def get_model(b, A):
    def line(x):
        return b * x + A
    return line


if __name__ == "__main__":

    model_params = pickle.load(
        open('../data/models/simple_regression.pkl', 'rb'))
    model = get_model(model_params[0], model_params[1])

    X = np.linspace(start=-1, stop=1, num=50)
    Ypred = [model(x) for x in X]
    plt.plot(X, Ypred)
    plt.title('Simple regression.')
    plt.ylabel('y predicted values')
    plt.xlabel('x values')
    plt.show()
