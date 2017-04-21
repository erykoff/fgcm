#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import argparse
import fgcm
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code to make FGCM look-up table')

    parser.add_argument('-c','--config', action='store', type=str, required=True,
                        help='YAML config file')
    parser.add_argument('-C','--clobber', action='store', type=bool, required=False,
                        default=False,help='Clobber existing LUT?')


    args = parser.parse_args()

    with open(args.config,'r') as f:
        lutConfig = yaml.load(f)

    if ((not args.clobber) and os.path.isfile(lutConfig['lutFile'])):
        print("LUT file %s already found, and clobber set to False." % (lutConfig['lutFile']))
        sys.exit(0)

    fgcmLUT = fgcm.FgcmLUT(lutConfig=lutConfig)
    fgcmLUT.makeLUT(lutConfig['lutFile'],clobber=True)
    fgcmLUT.makeLUTDerivatives(lutConfig['lutFile'])

