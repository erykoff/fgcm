#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import fgcm
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make FGCM stars from DES data')

    parser.add_argument('-c', '--config', action='store', type=str, required=True,
                        help='YAML config file')
    parser.add_argument('-C', '--clobber', action='store_true', required=False,
                        help='Clobber existing stars?')

    args = parser.parse_args()

    with open(args.config,'r') as f:
        starConfig = yaml.load(f)

    # INSERT CODE HERE TO MAKE THE OBSERVATION FILE FOR YOUR SURVEY
    # The output file should have the name starConfig['starfileBase']+'_observations.fits'

    # Next make the indexed and matched files
    fgcmMakeStars = fgcm.FgcmMakeStars(starConfig)
    fgcmMakeStars.runFromFits(clobber=args.clobber)
