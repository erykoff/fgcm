from __future__ import print_function

import os
import sys

from fgcmUtilities import logDict

class FgcmLogger(object):
    """
    """
    def __init__(self,logFile,logLevel):

        self.logFile = logFile

        # this might fail.  Let it throw its exception?
        self.logF = open(self.logFile,'w')

        self.logLevel = logLevel

        if (logLevel not in logDict):
            raise ValueError("Illegal logLevel: %s" % (logLevel))


    def log(self,logType,logString):

        if (logDict[logType] <= logDict[self.logLevel]):
            self.logF.write(logString)
            self.logF.flush()
            print(logString)

    def stopLogging(self):
        self.logF.close()

        self.logLevel == 0
