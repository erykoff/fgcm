import numpy as np
from multiprocessing import Lock, Manager
import multiprocessing
# from multiprocessing import shared_memory
# from multiprocessing.managers import SharedMemoryManager
import ctypes


class SharedNumpyMemHolder:
    def __init__(self):
        self._sharedArrayHandles = {}

    def setHandle(self, handleName, handle, dimensions, dtype, syncAccess):
        """
        """
        self._sharedArrayHandles[handleName] = (handle, dimensions, dtype, syncAccess)

    def getArray(self, handleName):
        """
        """
        if handleName not in self._sharedArrayHandles:
            raise ValueError("Unknown handle name %s.", handleName)

        handle, dimensions, dtype, _ = self._sharedArrayHandles[handleName]

        arr = np.frombuffer(handle.get_obj(), dtype=dtype)
        arr.reshape(dimensions)

        return arr

    def getArrayLock(self, handleName):
        """
        """
        if handleName not in self._sharedArrayHandles:
            raise ValueError("Unknown handle name %s.", handleName)

        handle, _, _, syncAccess = self._sharedArrayHandles[handleName]

        if not syncAccess:
            raise ValueError("No lock associated with %s.", handleName)

        return handle.get_lock()

    def getArrayHandle(self, handleName):
        """
        """
        if handleName not in self._sharedArrayHandles:
            raise ValueError("Unknown handle name %s.", handleName)

        return self._sharedArrayHandles[handleName][0]


class SharedNumpyMemManager2:

    def __init__(self):
        self._holder = SharedNumpyMemHolder()
        self._lock = Lock()
        # self._smm = SharedMemoryManager()
        # self._smm.start()
        self._manager = Manager()
        # self._manager.start()
        self._counter = 0

    def __del__(self):
        # self._smm.shutdown()
        # del self._smm
        self._manager.shutdown()
        del self._manager
        del self._lock
        del self._holder

    def createArray(self, dimensions, dtype=np.float64, syncAccess=False):
        """
        """
        dtype = np.dtype(dtype)
        if (dtype == np.float32):
            typecode = "f"
        elif (dtype == np.float64):
            typecode = "d"
        elif (dtype == np.int32):
            typecode = "i"
        elif (dtype == np.int64):
            typecode = "l"
        elif (dtype == np.int16):
            typecode = "h"
        elif (dtype == bool):
            typecode = "b"
        else:
            raise ValueError("Unsupported dtype")

        self._lock.acquire()

        # Must have locks ...
        # handle = self._manager.Array(typecode, range(int(np.prod(dimensions))))
        handle = multiprocessing.Array(typecode, int(np.prod(dimensions)))

        index = self._counter

        self._holder.setHandle(index, handle, dimensions, dtype, syncAccess)

        self._counter += 1

        self._lock.release()

        return index

    def createArrayLike(self, inArray, syncAccess=False, dtype=None):
        """
        """
        dimensions = inArray.shape
        if dtype is None:
            dtype = inArray.dtype

        return self._createArray(dimensions, dtype=dtype, syncAccess=syncAccess)

    def freeArray(self, handleName):
        """
        """
        self._lock.acquire()
        handle = self._holder._sharedArrayHandles.pop(handleName, None)
        if handle is not None:
            handle[0].unlink()

        self._lock.release()

    def getArray(self, handleName):
        """
        """
        return self._holder.getArray(handleName)

    def getArrayHandle(self, handleName):
        """
        """
        return self._holder.getArrayHandle(handleName)

    def getHolder(self):
        return self._holder
