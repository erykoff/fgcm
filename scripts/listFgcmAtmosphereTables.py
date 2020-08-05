#!/usr/bin/env python

import os
import sys
import argparse
import fgcm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code to list available FGCM atmosphere tables')

    parser.add_argument('-f', '--filename', action='store', type=str, required=False,
                        default=None, help='Atmosphere File')

    args = parser.parse_args()

    if args.filename is None:
        # list all of them
        availableTables = fgcm.FgcmAtmosphereTable.getAvailableTables()
    else:
        availableTables = {os.path.basename(args.filename):
                               fgcm.FgcmAtmosphereTable.getInfoDict(args.filename)}

    # and print it out all pretty like
    for fname in availableTables.keys():
        infoDict = availableTables[fname]
        print('Table name: '+ fname)
        print('  Elevation: %.2f' % (infoDict['elevation']))
        print('  PWV: %.2f - %.2f, %d steps, std = %.2f' %
              (infoDict['pwvRange'][0], infoDict['pwvRange'][1],
               infoDict['pwvSteps'], infoDict['pwvStd']))
        print('  Tau: %.3f - %.3f, %d steps, std = %.3f' %
              (infoDict['tauRange'][0], infoDict['tauRange'][1],
               infoDict['tauSteps'], infoDict['tauStd']))
        print('  Alpha: %.2f - %.2f, %d steps, std = %.3f' %
              (infoDict['alphaRange'][0], infoDict['alphaRange'][1],
               infoDict['alphaSteps'], infoDict['alphaStd']))
        print('  O3: %.2f - %.2f, %d steps, std = %.3f' %
              (infoDict['o3Range'][0], infoDict['o3Range'][1],
               infoDict['o3Steps'], infoDict['o3Std']))
        print('  Zenith: %.2f - %.2f, %d steps, airmass std = %.2f' %
              (infoDict['zenithRange'][0], infoDict['zenithRange'][1],
               infoDict['zenithSteps'], infoDict['airmassStd']))
        print('  PMB: %.2f - %.2f, %d steps, std = %.2f' %
              (infoDict['pmbRange'][0], infoDict['pmbRange'][1],
               infoDict['pmbSteps'], infoDict['pmbStd']))
        print('')
