import numpy as np
from threading import Lock

# Adapted from http://stackoverflow.com/questions/10721915/shared-memory-objects-in-python-multiprocessing

'''
Singleton Pattern
'''
class SharedNumpyMemManager(object):
    """
    Class to wrap numpy variables in shared memory for multiprocessing.

    Parameters
    ----------
    None

    Usage
    -----------
    # Import instantiates a singleton object to track arrays
    >>> from .sharedNumpyMemManager import SharedNumpyMemManager as snmm
    # Create a wrapped array of floats, size=100
    >>> arrayHandle = snmm.createArray(100, dtype='f4')
    # Get a pointer to the shared array
    >>> array = snmm.getArray(arrayHandle)
    # Set the array values
    >>> array[:] = 1

    Alternatively, you can create an array with a lock:
    # Create an array with an associated lock
    >>> lockArrayHandle = snmm.createArray(100, dtype='f4', syncAccess=True)
    # Get the lock associated with the array
    >>> lockArrayLock = snmm.getArraybase(lockArrayHandle).get_lock()
    # Acquire the lock
    >>> lockArrayLock.acquire()
    # Do work
    >>> lockArray = snmm.getArray(lockArrayHandle)E
    # Release lock
    >>> lockArrayLock.release()

    Note that the array is a pointer to the shared memory, so when
    run in multiprocessing the shared memory is not copied.

    Note that this can only wrap 1 or multi-dimensional numpy arrays,
    and cannot wrap objects or numpy recarrays, etc.

    Also note when creating wrapped arrays if you read in data
    then you need to clear this memory (setting to None) or else
    that input array will be copied in multiprocessing!
    """

    _initSize = 1024

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SharedNumpyMemManager, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        """ Create the singleton """

        self.lock = Lock()
        self.cur = 0
        self.cnt = 0
        self.sharedArrays = [None] * SharedNumpyMemManager._initSize

    def __createArrayLike(self, inArray, syncAccess=False, dtype=None):
        """
        Create an array with same format as another numpy array
        """
        # like zeros_like

        dimensions = inArray.shape
        if dtype is None:
            dtype = inArray.dtype

        return self.__createArray(dimensions, dtype=dtype, syncAccess=syncAccess)

    def __createArray(self, dimensions, dtype=np.float64, syncAccess=False):
        """
        Create an array
        """

        # This lock is only needed when creating the array, which is fast
        # It is not used when accessing data
        self.lock.acquire()

        # double size if necessary
        if (self.cnt >= len(self.sharedArrays)):
            self.sharedArrays = self.sharedArrays + [None] * len(self.sharedArrays)

        # next handle
        self.__getNextFreeHdl()

        # create array in shared memory segment
        if (syncAccess):
            try:
                self.sharedArrays[self.cur] = (np.zeros(dimensions, dtype=dtype), Lock())
            except:
                raise MemoryError("Failed to allocate memory for shared array.")
        else:
            try:
                self.sharedArrays[self.cur] = (np.zeros(dimensions, dtype=dtype), None)
            except:
                raise MemoryError("Failed to allocate memory for shared array.")


        # update cnt
        self.cnt += 1

        # Release the creation lock
        self.lock.release()

        # return handle to the shared memory numpy array
        return self.cur

    def __getNextFreeHdl(self):
        """
        Get the next free handle
        """
        orgCur = self.cur
        while self.sharedArrays[self.cur] is not None:
            self.cur = (self.cur + 1) % len(self.sharedArrays)
            if orgCur == self.cur:
                raise SharedNumpyMemManagerError('Max Number of Shared Numpy Arrays Exceeded!')

    def __freeArray(self, hdl):
        """
        Free an array
        """
        self.lock.acquire()
        # set reference to None
        if self.sharedArrays[hdl] is not None: # consider multiple calls to free
            self.sharedArrays[hdl] = None
            self.cnt -= 1
        self.lock.release()

    def __getArray(self, i):
        return self.sharedArrays[i]

    @staticmethod
    def getInstance():
        if not SharedNumpyMemManager._instance:
            SharedNumpyMemManager._instance = SharedNumpyMemManager()
        return SharedNumpyMemManager._instance

    @staticmethod
    def createArray(*args, **kwargs):
        """
        Create an wrapped numpy array

        Parameters
        ----------
        dimensions: scalar or tuple
           Numpy array dimensions
        dtype: numpy dtype, default float64
           float32, float64, int16, int32, int64, or bool
        syncAccess: bool, default False
           Associate array with lock
        """
        return SharedNumpyMemManager.getInstance().__createArray(*args, **kwargs)

    @staticmethod
    def createArrayLike(*args, **kwargs):
        """
        Create an array like another numpy array, as with np.array_like()

        Parameters
        ----------
        array: numpy array
           Numpy array to duplicate type
        syncAccess: bool, default false
           Associate array with lock
        dtype: numpy dtype
           Override array dtype with float32, float64, int16, int32, int64, or bool
        """
        return SharedNumpyMemManager.getInstance().__createArrayLike(*args, **kwargs)

    @staticmethod
    def getArray(*args, **kwargs):
        """
        Get a reference to an array from a handle

        Parameters
        ----------
        handle: integer
           Array handle

        Returns
        -------
        Reference to numpy array
        """
        return SharedNumpyMemManager.getInstance().__getArray(*args, **kwargs)[0]

    @staticmethod
    def getArrayLock(*args, **kwargs):
        return SharedNumpyMemManager.getInstance().__getArray(*args, **kwargs)[1]

    @staticmethod
    def freeArray(*args, **kwargs):
        """
        Free wrapped array

        Parameters
        ----------
        handle: integer
           Array handle
        """
        return SharedNumpyMemManager.getInstance().__freeArray(*args, **kwargs)

# Init Singleton on module load
SharedNumpyMemManager.getInstance()


