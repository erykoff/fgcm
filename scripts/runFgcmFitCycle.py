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

    args = parser.parse_args()

    #fgcmFitCycle = fgcm.FgcmFitCycle(args.config)
    #fgcmFitCycle = fgcm.FgcmFitCycleNew(args.config)
    #fgcmFitCycle.run()

    with open(args.config) as f:
        configDict = yaml.load(f)

    print("Configuration read from %s" % (args.config))

    fgcmFitCycle = fgcm.FgcmFitCycle(configDict, useFits=True)
    fgcmFitCycle.runWithFits()
    

