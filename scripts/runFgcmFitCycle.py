#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import argparse
import fgcm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code to run FGCM fit cycle')

    parser.add_argument('-c','--config', action='store', type=str, required=True,
                        help='YAML config file')

    args = parser.parse_args()

    fgcmFitCycle = fgcm.FgcmFitCycle(args.config)

    fgcmFitCycle.run()

