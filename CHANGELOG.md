# All notable changes will be documented in this file

## Unreleased

Changes currently on master, but not under a tag.

- Removed unnecessary arguments (`ra`, `dec`, `geocent_time`, `psi`) from source functions and replaced them with `**kwargs` where appropriate.
- Renamed `PriorSet` to `PriorDict`
- Renamed `BBHPriorSet` to `BBHPriorDict`
- Renamed `BNSPriorSet` to `BNSPriorDict`
- Renamed `CalibrationPriorSet` to `CalibrationPriorDict`
- Fixed a bug which caused `Interferometer.detector_tensor` not to update when `latitude`, `longitude`, `xarm_azimuth`, `yarm_azimuth`, `xarm_tilt`, `yarm_tilt` were updated.

## [0.3.1] 2018-11-06

### Changes
- Make BBH/BNS parameter conversion more logical
- Source frame masses/spins included in posterior
- Make filling in posterior with fixed parameters work
- Bug fixes

## [0.3] 2018-11-02

### Added
- Joint-likelihood added
- PyMC3 works with the GravitationalWaveTransient likelihood
- flake8 syntax checking in CI
- Binary neutron star source model
- Allow units to be included in parameter labels
- Add nested samples to dynesty output
- Add more \_\_repr\_\_ methods
- Add ability to plot max likelihood and draws from the posterior
- Document samplers in more detail
- Added the CPNest sampler
- Adds custom titles to corner plots
- Adds plotting of the prior on 1D marginal distributions of corner plots
- Adds a method to plot time-domain GW data
- Added pipenv as a dependency manager
- Hyperparameter estimation now enables the user to provide the single event evidences
- Add nested samples to nestle output
- Prior and child classes now implement the \_\_eq\_\_ magic method for comparisons
- Added default kwargs for each sampler class
- Added NestedSampler and MCSampler helper classes
- Added sampler_requirements.txt file
- Add AlignedSpin gw prior
- Add units to know prior files
- Add pipenv functionality
- Tests run in custom dockerfiles

### Changes
- Fix construct_cbc_derived_parameters
- Autocorrelation calculation moved into parent class
- Fix interpretation of kwargs for dynesty
- PowerSpectralDensity structure modified
- Fixed bug in get_open_data
- Modified how sampling in non-standard parameters is done, the
  `non_standard_sampling_parameter_keys` kwarg has been removed
- .prior files are no longer created. The prior is stored in the result object.
- Removed external_sampler and external_sampler_function attribute from Sampler
- Made conversion of number of livepoint kwargs consistent and streamlined throughout the Nested sampler classes
- Fix label creation in plot_multiple, evidences and repeated plots.
- Changed the way repr works for priors. The repr can now be used to
re-instantiate the Prior in most cases
- Users can now choose to overwrite existing result files, rather than creating
  a .old file.
- Make likelihood values stored in the posterior correct for dynesty and nestle
- pymultinest output now stored in {outdir}/pm_{label}/

### Removed
- Removes the "--detectors" command line argument (not a general CLI requirement)

## [0.2.2] 2018-09-04

### Added
- Add functionality to sample in redshift and reconstruction of source frame masses.
- Add functionality to combine result objects.
- Enable initial values for emcee to be specified.
- Add support for eccentric BBH

### Changed
- Specifying detectors by name from the default command line options has been removed.
- The prior on polarisation phase has been reduced to [0, pi].
- More prior distributions added.
- More samplers supported, pymc3
- More core likelihoods, Poisson, Student's-t
- Result print function fixed
- Add snr functions as methods of `Interferometer`
- The paths between imports where changed so that calls such as
  `bilby.WaveformGenerator` no longer work. Instead, we need to use
  `bilby.gw.WaveformGenerator`. This was done to keep things cleaner going
  forward (when, for example, there may be multiple wfg's).
- Samplers reorganised into individual files.

## [0.2.1] 2018-07-18

### Added
- InterferometerStrainData now handles both time-domain and frequencu-domain data
- Adds documentation on setting data (https://monash.docs.ligo.org/bilby/transient-gw-data.html)
- Checkpointing for `dynesty`: the sampling will be checkpointed every 10 minutes (approximately) and can be resumed.
- Add functionality to plot multiple results in a corner plot, see `bilby.core.result.plot_multiple()`.
- Likelihood evaluations are now saved along with the posteriors.

### Changed
- Changed to using `setuptools` for installation.
- Clean up of real data handling: all data is now windowed with a 0.4s roll off (unless set otherwise) and low-pass filtered.
- Add explicit method to create a power spectral density from time-domain data
- Clean up of `PowerSpectralDensity()` - addds `set_from` methods to handle various ways to define the PSD.
- Clean up of `detectors.py`: adds an `InterferometerStrainData` to handle strain data and `InterferometerSet` to handle multiple interferometers. All data setting should primarily be done through the `Interferometer.set_strain_data..` methods.
- Fix the comments and units of `nfft` and `infft` and general improvement to documentation of data.
- Fixed a bug in create_time_series
- Enable marginalisation over calibration uncertainty in Inteferemeter data.
- Fixed the normalisation of the marginalised `GravtitationalWaveTransient` likelihood.
- Fixed a bug in the detector response.

## [0.2.0] 2018-06-17

First `pip` installable version https://pypi.org/project/BILBY/ .

### Added
- Reoriganisation of the directory into `bilby.core` and `bilby.gw`.
- Reading of frame files.
- Major effort to update all docstrings and add some documentation.
- Marginalized likelihoods.
- Examples of searches for gravitational waves from a Supernova and using a sine-Gaussian.
- A `PriorSet` to handle sets of priors and allows reading in from a standardised prior file (see https://monash.docs.ligo.org/bilby/prior.html).
- A standardised file for storing detector data.

### Removed
- All chainconsumer dependency as this was causing issues.

