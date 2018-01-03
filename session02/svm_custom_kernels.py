
import numpy as np

def intersection_kernel(A, B):

        kernel = np.zeros((A.shape[0], B.shape[0]))

        for i in range(A.shape[1]):
            column_A = A[:, i].reshape(-1, 1)
            column_B = B[:, i].reshape(-1, 1)
            kernel += np.minimum(column_A, column_B.T)

        return kernel
