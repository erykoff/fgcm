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

Note that the "standard atmosphere" parameters are chosen for each table.  A
new table will have to be generated to get a new standard atmosphere (different
reference airmass, for example).

## Ingesting and Matching Star Observations

First, you need to create an observation file which is a giant fits file with
all the observations to be used in the calibration.  This can be tens of
gigabytes.  In the file should be:

```
'FILTERNAME': Name of the filter used
'RA': RA
'DEC': Dec
'MAG': raw 'ADU' exposure-time corrected magnitude, MAG = -2.5*log10(FLUX)
+ 2.5*log10(EXPTIME)
'MAGERR': magnitude error
expField: Exposure/visit number, name is configurable
ccdField: ccd number, name is configurable
'X': x position on CCD (optional)
'Y': y position on CCD (optional)
```

The name of the file should be of the form `starfileBase+'_observations.fits`
for input into the matching code.

Next, you can run the matching code.  See `examples/scripts/makeStarsSample.py`
and `examples/configs/fgcm_sample_stars.yml`.

## Running a Fit Cycle

### The First Cycle (Cycle 0)

See `examples/scripts/fgcm_sample_cycle00_config.yml` for a sample config file
and an explanation of what is required for inputs in each file.  In addition to
the LUT and star files, tables describing exposure parameters (including MJD,
telescope pointing, and barometric pressure), CCD offset positions (for airmass
corrections for the final zeropoints for large cameras, and for plotting).  You
also need to define observational epochs (preferably longish timescales when
new flats were generated), and input dates (in MJD units) when the mirror was
washed/recoated.

There are two important differences between the first cycle and subsequent
cycles.  The first is that a "bright observation" algorithm is employed to
choose approximately photometric observations.  The second is that I recommend
that you freeze the atmosphere to the standard parameters for the first fit
cycle.

At the end of the first (and all subsequent) cycles a bunch of diagnostic plots
are made in a subdirectory generated from the `outfileBase` and the
`cycleNumber`.  In addition, a new config file is output for the next cycle
that automatically increments the `cycleNumber` and turns off
`freezeStdAtmosphere`.

### Subsequent Cycles

Before running any subsequent cycle, you should especially look at the
`cycleNUMBER_expgray_BAND.png` plots.  Choose appropriate cuts for each band to
select "photometric" exposures in the next cycle with the
`expGrayPhotometricCut` and `expGrayHighCut` parameters.  You can also up the
number of iterations per cycle.  In my experience, the fit does not improve if
you go beyond ~50 iterations.  The best way to get the fit to improve is to
remove non-photometric exposures.

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
* hpgeom
* pyyaml
* smatch (optional, for the fastest object matching)
* mpl_toolkits (optional, for fancy coverage map plotting)
