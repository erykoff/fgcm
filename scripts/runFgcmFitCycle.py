#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import argparse
import fgcm
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code to run FGCM fit cycle')

    parser.add_argument('-c','--config', action='store', type=str, required=True,
                        help='YAML config file')
    parser.add_argument('-C','--clobber', action='store_true', default=False, help='Clobber existing run')
    parser.add_argument('-p','--printOnly', action='store_true', default=False, help='Only print logging to stdout')

    args = parser.parse_args()

    with open(args.config) as f:
        configDict = yaml.load(f)

    print("Configuration read from %s" % (args.config))

    configDict['clobber'] = args.clobber
    configDict['printOnly'] = args.printOnly

    fgcmFitCycle = fgcm.FgcmFitCycle(configDict, useFits=True)
    fgcmFitCycle.runWithFits()


