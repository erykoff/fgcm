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
                        default=False,help='Clobber existing stars?')

    args = parser.parse_args()

    with open(args.config,'r') as f:
        starConfig = yaml.load(f)

    fgcmMakeStars = fgcm.FgcmMakeStars(starConfig)
    fgcmMakeStars.run(clobber=args.clobber)
    
