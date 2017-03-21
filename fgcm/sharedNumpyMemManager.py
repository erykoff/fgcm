import numpy as np
import multiprocessing

import ctypes

from fgcmUtilities import _pickle_method
import types
import copy_reg


copy_reg.pickle(types.MethodType, _pickle_method)

# Adapted from http://stackoverflow.com/questions/10721915/shared-memory-objects-in-python-multiprocessing

'''
Singleton Pattern
'''
class SharedNumpyMemManager:

    _initSize = 1024

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SharedNumpyMemManager, cls).__new__(
                                cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.lock = multiprocessing.Lock()
        self.cur = 0
        self.cnt = 0
        self.shared_arrays = [None] * SharedNumpyMemManager._initSize

    #def __createArray(self, dimensions, ctype=ctypes.c_double, dtype=None):
    def __createArray(self, dimensions, dtype=np.float64, syncAccess=False):
        # convert to dtype type (in case short code)
        dtype = np.dtype(dtype)
        if (dtype == np.float32):
            ctype = ctypes.c_float
        elif (dtype == np.float64):
            ctype = ctypes.c_double
        elif (dtype == np.int32):
            ctype = ctypes.c_int32
        elif (dtype == np.int64):
            ctype = ctypes.c_int64
        elif (dtype == np.int16):
            ctype = ctypes.c_int16
        elif (dtype == np.bool):
            ctype = ctypes.c_bool
        else:
            raise ValueError("Unsupported dtype")

        self.lock.acquire()

        # double size if necessary
        if (self.cnt >= len(self.shared_arrays)):
            self.shared_arrays = self.shared_arrays + [None] * len(self.shared_arrays)

        # next handle
        self.__getNextFreeHdl()

        # create array in shared memory segment
        if (syncAccess):
            shared_array_base = multiprocessing.Array(ctype, np.prod(dimensions))
        else:
            shared_array_base = multiprocessing.RawArray(ctype, np.prod(dimensions))

        # convert to numpy array vie ctypeslib
        if (syncAccess):
            self.shared_arrays[self.cur] = np.frombuffer(shared_array_base.get_obj(),dtype=dtype)
        else:
            self.shared_arrays[self.cur] = np.frombuffer(shared_array_base,dtype=dtype)


        # do a reshape for correct dimensions
        # Returns a masked array containing the same data, but with a new shape.
        # The result is a view on the original array
        self.shared_arrays[self.cur] = self.shared_arrays[self.cnt].reshape(dimensions)

        # update cnt
        self.cnt += 1

        self.lock.release()

        # return handle to the shared memory numpy array
        return self.cur

    def __getNextFreeHdl(self):
        orgCur = self.cur
        while self.shared_arrays[self.cur] is not None:
            self.cur = (self.cur + 1) % len(self.shared_arrays)
            if orgCur == self.cur:
                raise SharedNumpyMemManagerError('Max Number of Shared Numpy Arrays Exceeded!')

    def __freeArray(self, hdl):
        self.lock.acquire()
        # set reference to None
        if self.shared_arrays[hdl] is not None: # consider multiple calls to free
            self.shared_arrays[hdl] = None
            self.cnt -= 1
        self.lock.release()

    def __getArray(self, i):
        return self.shared_arrays[i]

    @staticmethod
    def getInstance():
        if not SharedNumpyMemManager._instance:
            SharedNumpyMemManager._instance = SharedNumpyMemManager()
        return SharedNumpyMemManager._instance

    @staticmethod
    def createArray(*args, **kwargs):
        return SharedNumpyMemManager.getInstance().__createArray(*args, **kwargs)

    @staticmethod
    def getArray(*args, **kwargs):
        return SharedNumpyMemManager.getInstance().__getArray(*args, **kwargs)

    @staticmethod
    def freeArray(*args, **kwargs):
        return SharedNumpyMemManager.getInstance().__freeArray(*args, **kwargs)

# Init Singleton on module load
SharedNumpyMemManager.getInstance()


