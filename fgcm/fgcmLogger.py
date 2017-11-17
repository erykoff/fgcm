from __future__ import print_function

import os
import sys

from fgcmUtilities import logDict

class FgcmLogger(object):
    """
    """
    def __init__(self,logFile,logLevel,printLogger=False):

        self.logFile = logFile
        self.printLogger = printLogger

        if not printLogger:
            # this might fail.  Let it throw its exception?
            self.logF = open(self.logFile,'w')

        self.logLevel = logLevel

        if (logLevel not in logDict):
            raise ValueError("Illegal logLevel: %s" % (logLevel))

    def info(self,logString):
        """
        """
        self.log('INFO', logString)

    def debug(self,logString):
        """
        """
        self.log('DEBUG', logString)

    def log(self,logType,logString):
        """
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

    def logMemoryUsage(self,logType,location):
        """
        """
        status = None
        result = {'peak':0, 'rss':0}
        try:
            status = open('/proc/self/status')
            for line in status:
                parts = line.split()
                key = parts[0][2:-1].lower()
                if key in result:
                    result[key] = int(parts[1])/1000

            logString = 'Memory usage at %s: %d MB current; %d MB peak.' % (
                location, result['rss'], result['peak'])

            self.log(logType,logString)
        except:
            self.log(logType,'Could not get process status for resource usage!')

        return

    def stopLogging(self):
        """
        """
        if not self.printLogger:
            self.logF.close()

        self.logLevel == 0
