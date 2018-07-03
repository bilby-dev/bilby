# All notable changes will be documented in this file

## Unreleased

Changes currently on master, but not under a tag.

### Added
- InterferometerStrainData now handles both time-domain and frequencu-domain data
- Adds documentation on setting data (https://monash.docs.ligo.org/tupak/transient-gw-data.html)
- Checkpointing for `dynesty`: the sampling will be checkpointed every 10 minutes (approximately) and can be resumed.
- Add functionality to plot multiple results in a corner plot, see `tupak.core.result.plot_multiple()`.
- Likelihood evaluations are now saved along with the posteriors.

### Changed
- Clean up of real data handling: all data is now windowed with a 0.4s roll off (unless set otherwise) and low-pass filtered.
- Add explicit method to create a power spectral density from time-domain data
- Clean up of `PowerSpectralDensity()` - addds `set_from` methods to handle various ways to define the PSD.
- Clean up of `detectors.py`: adds an `InterferometerStrainData` to handle strain data and `InterferometerSet` to handle multiple interferometers. All data setting should primarily be done through the `Interferometer.set_strain_data..` methods.
- Fix the comments and units of `nfft` and `infft` and general improvement to documentation of data.
- Fixed a bug in create_time_series

## [0.2.0] 2018-06-17

First `pip` installable version https://pypi.org/project/TUPAK/ .

### Added
- Reoriganisation of the directory into `tupak.core` and `tupak.gw`.
- Reading of frame files.
- Major effort to update all docstrings and add some documentation.
- Marginalized likelihoods.
- Examples of searches for gravitational waves from a Supernova and using a sine-Gaussian.
- A `PriorSet` to handle sets of priors and allows reading in from a standardised prior file (see https://monash.docs.ligo.org/tupak/prior.html).
- A standardised file for storing detector data.

### Removed
- All chainconsumer dependency as this was causing issues.

