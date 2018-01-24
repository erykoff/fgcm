Forward Global Calibration Method (FGCM)
========================================

Code to run the Forward Global Calibration Method (FGCM) for survey calibration
including chromatic corrections.  See
http://adsabs.harvard.edu/abs/2018AJ....155...41B for details on the
algorithm. This code should be compatible with any survey with the following
requirements, and has been run on DES and HSC data (via the LSST stack):

* Exposure/CCD based observations
* Transmission curves for each filter, preferably for each CCD
* MODTRAN4 is required to generate new atmosphere tables.
* Atmosphere tables for the following telescope locations are included:
    - Blanco telescope at CTIO (DES)
    - Subaru telescope at Mauna Kea (HSC)
    - LSST telescope at Cerro Pachon (LSST)
* Enough memory to hold all the observations in memory at once.
    - A full run of four years of DES survey data can be run on a machine
      with 128 Gb RAM and 32 processors in less than a day.

Installation
------------

You can install this module with
```
cd fgcm
python setup.py install
```

Documentation
-------------

(The documentation is incomplete.)

There are three stages to running an FGCM calibration on a dataset:

1. Construct an atmosphere and instrument look-up table (LUT)
2. Ingest and match the individual star observations from exposure/ccd sets.
3. Run the calibration, with 25-100 fit iterations per "fit cycle".

## Constructing a Look-Up Table (LUT)

The script `listFGCMAtmosphereTables.py` will list what tables are available.
Feel free to ask for another if you don't have MODTRAN4 available.

Actually making the LUT requires not only the atmosphere table, but a model for
the instrumental throughput as a function of CCD (due to filter and QE
variations).  See `examples/scripts/makeLUTSample.py` and
`examples/configs/fgcm_sample_lut.yml` for a framework on how to
make a LUT.

## Ingesting and Matching Star Observations

## Running a Fit Cycle

### The First Cycle (Cycle 0)

### Subsequent Cycles

Tests
-----

Public tests to be written (sorry).

Dependencies
------------

A list of dependencies includes the following:

* numpy
* scipy
* matplotlib
* esutil
* fitsio (for standalone running)
* healpy
* pyyaml
* mpl_toolkits (for fancy coverage map plotting)
