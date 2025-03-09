import numpy as np
from multiprocessing import Lock, Manager
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager


class SharedNumpyMemHolder:
    def __init__(self):
        self._sharedArrayHandles = {}

    def setHandle(self, handle, dimensions, dtype, lock=None):
        """
        """
        self._sharedArrayHandles[handle.name] = (handle, dimensions, dtype, lock)

    def getArray(self, handleName):
        """
        """
        if handleName not in self._sharedArrayHandles:
            raise ValueError("Unknown handle name %s.", handleName)

        handle, dimensions, dtype, _ = self._sharedArrayHandles[handleName]

        return np.ndarray(dimensions, dtype=dtype, buffer=handle.buf)

    def getArrayLock(self, handleName):
        """
        """
        if handleName not in self._sharedArrayHandles:
            raise ValueError("Unknown handle name %s.", handleName)

        return self._sharedArrayHandles[handleName][3]

    def getArrayHandle(self, handleName):
        """
        """
        if handleName not in self._sharedArrayHandles:
            raise ValueError("Unknown handle name %s.", handleName)

        if handleName not in self._sharedArrayHandles:
            raise valueError("Unknown handle name %s.", handleName)

        return self._sharedArrayHandles[handleName][0]


class SharedNumpyMemManager2:

    def __init__(self):
        self._holder = SharedNumpyMemHolder()
        self._lock = Lock()
        self._smm = SharedMemoryManager()
        self._smm.start()
        self._manager = Manager()

    def __del__(self):
        self._smm.shutdown()
        del self._smm
        del self._lock
        del self._holder

    def createArray(self, dimensions, dtype=np.float64, syncAccess=False):
        """
        """
        dtype = np.dtype(dtype)

        self._lock.acquire()

        handle = self._smm.SharedMemory(int(np.prod(dimensions))*dtype.itemsize)

        if syncAccess:
            handleLock = self._manager.Lock()
        else:
            handleLock = None

        self._holder.setHandle(handle, dimensions, dtype, lock=handleLock)

        self._lock.release()

        return handle.name

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
