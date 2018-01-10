from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import os
import sys

from .fgcmUtilities import logDict

class FgcmLogger(object):
    """
    Class to do logging for FGCM.  The interface here is the same as the LSST DM
     logger so that logger can be dropped right in and used instead.

    parameters
    ----------
    logFile: string
       File to write log to
    logLevel: string
       Set to 'INFO' or 'DEBUG'
    printLogger: bool, default=False
       Only print log messages to stdout, and do not open logFile
    """

    def __init__(self,logFile,logLevel,printLogger=False):

        self.logFile = logFile
        self.printLogger = printLogger
        self.logging = False

        if not printLogger:
            # this might fail.  Let it throw its exception?
            self.logF = open(self.logFile,'w')
            self.logging = True

        self.logLevel = logLevel

        if (logLevel not in logDict):
            raise ValueError("Illegal logLevel: %s" % (logLevel))

    def pause(self):
        """
        Pause logging for multiprocessing
        """

        if self.logging:
            self.logF.close()
            self.logF = None
            self.logging = False

    def resume(self):
        """
        Resume logging for multiprocessing
        """

        if not self.logging and not self.printLogger:
            self.logF = open(self.logFile, "a")
            self.logging = True

    def info(self,logString):
        """
        Log at info level.

        parameters
        ----------
        logString: string
           String to write to log
        """

        self.log('INFO', logString)

    def debug(self,logString):
        """
        Log at debug level.

        parameters
        ----------
        logString: string
           String to write to log
        """

        self.log('DEBUG', logString)

    def log(self,logType,logString):
        """
        Log at any level.

        parameters
        ----------
        logType: string
           Level at which to log.  'INFO' or 'DEBUG'
        logString: string
           String to write to log
        """

        if (logDict[logType] <= logDict[self.logLevel]):
            if not self.printLogger:
                self.logF.write(logString+'\n')
                self.logF.flush()

            try:
                # we also print to stdout
                print(logString)
            except:
                # sometimes it just fails...let it go.
                pass

    def stopLogging(self):
        """
        Stop logging and close the file.
        """

        if not self.printLogger:
            self.logF.close()

        self.logLevel == 0
