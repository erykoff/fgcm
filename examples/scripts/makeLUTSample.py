#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import fgcm
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample framework to make FGCM look-up table')

    parser.add_argument('-c','--config', action='store', type=str, required=True,
                        help='YAML config file')
    parser.add_argument('-C','--clobber', action='store_true',
                        default=False,help='Clobber existing LUT?')
    parser.add_argument('-s','--makeSeds', action='store_true',
                        default=False,help='Make SEDs?')

    args = parser.parse_args()

    with open(args.config,'r') as f:
        lutConfig = yaml.load(f)

    if 'lutFile' not in lutConfig:
        raise ValueError("Must include lutFile in config to run makeFgcmLUT.py")

    if ((not args.clobber) and os.path.isfile(lutConfig['lutFile'])):
        print("LUT file %s already found, and clobber set to False." % (lutConfig['lutFile']))
        sys.exit(0)

    # makeSeds option will optionally generate star color -> SED slope look-up table.
    # Using this SED LUT slightly degrades the calibration, and I'm not sure why.

    fgcmLUTMaker = fgcm.FgcmLUTMaker(lutConfig, makeSeds=args.makeSeds)

    # Insert here code to generate a throughputDict based on your instrument.
    # This is a dictionary of dictionaries like:
    #  throughputDict = {'g': tDictg, 'r': tDictr} # etc
    # Each tDict has the following keys:
    #  tDictg = {'LAMBDA': np.array(wavelengths),
    #            'AVG': np.array(focal-plane average)
    #            ccdIndex0: np.array(ccd0 throughput)
    #            ccdIndex1: np.array(ccd1 throughput)}
    #            ...etc..
    # The AVG key is optional, and the code will compute a focal-plane average from
    # the ccds if necessary.
    # Note that the CCDs are zero-indexed and sequential.  I suppose this should
    # be made more transparent.

    fgcmLUTMaker.setThroughputs(throughputDict)
    fgcmLUTMaker.makeLUT()
    fgcmLUTMaker.saveLUT(args.lutFile, clobber=args.clobber)

