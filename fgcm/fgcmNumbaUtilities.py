from __future__ import division, absolute_import, print_function

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

    @jit
    def add_at(array, indices, values):
        if isinstance(indices, tuple):
            ctr = 0
            for ind in zip(*indices):
                array[ind] += values[ctr]
                ctr += 1
        else:
            for i in range(indices.size):
                array[indices[i]] += values[i]

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

    @jit
    def numba_test(value):
        print("Using numba")
else:
    import numpy as np

    add_at_single = np.add.at
    add_at = np.add.at
    add_at_2d = np.add.at
    add_at_1d = np.add.at
    add_at_3d = np.add.at
    add_at_3d_single = np.add.at

    def numba_test(value):
        print("Not using numba")


