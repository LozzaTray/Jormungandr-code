from inference.softmax import SoftmaxNeuralNet
import numpy as np


def run():
    B = 2
    D = 4
    classifier = SoftmaxNeuralNet([D, B], sigma=100)
    X = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    Y = np.array([[0.8, 0.2], [0.2, 0.8]])

    #classifier.fit(X, Y, verbose=True)
    classifier.sgld_initialise()

    for n in range(0, 1000):
        cost = classifier.sgld_iterate(X, Y)
        if n % 100 == 0:
            print("cost:{}".format(cost))


if __name__ == "__main__":
    print("Testing softmax class")
    run()