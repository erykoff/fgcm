from __future__ import print_function

import os
import sys

from fgcmUtilities import logDict

class FgcmLogger(object):
    """
    """
    def __init__(self,logFile,logLevel,printOnly=False):

        self.printOnly = printOnly

        self.logFile = logFile

        if not self.printOnly:
            # this might fail.  Let it throw its exception?
            self.logF = open(self.logFile,'w')

        self.logLevel = logLevel

        if (logLevel not in logDict):
            raise ValueError("Illegal logLevel: %s" % (logLevel))


    def log(self,logType,logString,printOnly=False):
        """
        """

        if (logDict[logType] <= logDict[self.logLevel]):
            if (not printOnly and not self.printOnly):
                self.logF.write(logString+'\n')
                self.logF.flush()

            try:
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
        if not self.printOnly:
            self.logF.close()

        self.logLevel == 0
