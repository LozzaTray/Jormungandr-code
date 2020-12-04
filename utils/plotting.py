import matplotlib.pyplot as plt
import numpy as np


def sorted_bar_plot(values, labels, title):
    sorted_indices = np.argsort(np.abs(values))
    sorted_values = values[sorted_indices]
    sorted_labels = labels[sorted_indices]

    plt.figure()
    plt.barh(sorted_labels, sorted_values)
    plt.title(title)
    plt.grid()
    plt.show()