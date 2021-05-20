import numpy as np


def random_index_arr(n, fraction):
    """
    Return two index arrays to form test and training set

        Parameters:
            n (int): length of array to index
            fraction (float): fraction to use for training set

        Returns:
            train_indices (int[]): indices of training set
            test_indices (int[]): indices of test set    
    """
    indices = np.random.permutation(n)
    m = int(n * fraction)
    train_indices = indices[:m]
    test_indices = indices[m:]
    return train_indices, test_indices