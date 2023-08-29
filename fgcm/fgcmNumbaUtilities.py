try:
    from numba import jit
    has_numba = True
except ImportError:
    has_numba = False


if has_numba:
    @jit(nopython=True)
    def add_at_single(array, indices, value):
        for ind in indices:
            array[ind] += value

    @jit(nopython=True)
    def add_at_1d(array, indices, values):
        for i in range(indices.size):
            array[indices[i]] += values[i]

    @jit(nopython=True)
    def add_at_2d(array, indices, values):
        for i in range(indices[0].size):
            array[indices[0][i], indices[1][i]] += values[i]

    @jit(nopython=True)
    def add_at_3d(array, indices, values):
        for i in range(indices[0].size):
            array[indices[0][i], indices[1][i], indices[2][i]] += values[i]

    @jit(nopython=True)
    def add_at_3d_single(array, indices, value):
        for i in range(indices[0].size):
            array[indices[0][i], indices[1][i], indices[2][i]] += value

    @jit(nopython=True)
    def numba_test(value):
        pass

else:
    import numpy as np

    add_at_single = np.add.at
    add_at = np.add.at
    add_at_2d = np.add.at
    add_at_1d = np.add.at
    add_at_3d = np.add.at
    add_at_3d_single = np.add.at

    def numba_test(value):
        pass


