#!/usr/bin/env python

import os
import sys
import argparse
import fgcm
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code to make FGCM modtran atmosphere tables')

    parser.add_argument('-c','--config', action='store', type=str, required=True,
                        help='YAML config file')
    parser.add_argument('-C','--clobber', action='store', type=bool, required=False,
                        default=False,help='Clobber existing LUT?')


    args = parser.parse_args()

    with open(args.config,'r') as f:
        lutConfig = yaml.load(f, Loader=yaml.SafeLoader)

    if 'atmosphereTableFile' not in lutConfig:
        raise ValueError("Must include atmosphereTableFile in config to run makeFgcmAtmosphereTable.py")

    if ((not args.clobber) and os.path.isfile(lutConfig['atmosphereTableFile'])):
        print("atmosphereTableFile %s already found, and clobber set to False." % (lutConfig['atmosphereTableFile']))
        sys.exit(0)

    fgcmAtmosphereTable = fgcm.FgcmAtmosphereTable(lutConfig)
    fgcmAtmosphereTable.generateTable()
    fgcmAtmosphereTable.saveTable(lutConfig['atmosphereTableFile'])



