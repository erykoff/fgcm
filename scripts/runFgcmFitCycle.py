#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import matplotlib
matplotlib.use("Agg")  # noqa E402

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
        configDict = yaml.load(f, Loader=yaml.SafeLoader)

    print("Configuration read from %s" % (args.config))

    configDict['clobber'] = args.clobber
    configDict['printOnly'] = args.printOnly

    fgcmFitCycle = fgcm.FgcmFitCycle(configDict, useFits=True)
    fgcmFitCycle.runWithFits()


