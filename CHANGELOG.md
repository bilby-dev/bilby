# All notable changes will be documented in this file

## Unreleased

### Added
- 

### Changed
- 

### Removed
-

## [0.4.4] 2019-04-03

### Added
- Infrastucture for custom jump proposals (cpnest-only)
- Evidence uncertainty estimate to cpnest

### Changed
- Bug fix to close figures after creation
- Improved the frequency-mask to entirely remove values outside the mask rather
  than simply set them to zero
- Fix problem with Prior prob and ln_prob if passing multiple samples
- Improved cpnest prior sampling

### Removed
-

## [0.4.3] 2019-03-21

### Added
- Constraint prior: in prior files you can now add option of a constraint based
on other parameters. Currently implements mass-constraints only.
- Grid likelihood: module to evaluate the likelihood on a grid

### Changed
- The GWTransientLikelihood no longer returns -inf for  m2 > m1. It will evaluate
the likelihood as-is. To implement the constraint, use the Constraint priors.

## [0.4.2] 2019-03-21

### Added
- Fermi-Dirac and SymmetricLogUniform prior distributions
- Multivariate Gaussian example and BNS example
- Added standard GWOSC channel names
- Initial work on a fake sampler for testing
- Option for aligned spins
- Results file command line interface
- Full reconstruction of marginalized parameters

### Changed
- Fixed scheduled tests and simplify testing environment
- JSON result files can now be gzipped
- Reduced ROQ memory usage
- Default checkpointing in cpnest

## [0.4.1] 2019-03-04

### Added
- Support for JSON result files
- Before sampling a test is performed for redundant priors

### Changed
- Fixed the definition of iota to theta_jn. WARNING: this breaks backward compatibility. Previously, the CBC parameter iota was used in prior files, but was ill-defined. This fixes that, requiring all scripts to use `theta_jn` in place of `iota`
- Changed the default result file store to JSON rather than hdf5. Reading/writing of hdf5 files is still intact. The read_in_result function will still read in hdf5 files for backward compatibility
- Minor fixes to the way PSDs are calculated
- Fixed a bug in the CBC result where the frequency_domain model was pickled
- Use pickling to store the dynesty resume file and add a write-to-resume on SIGINT/SIGKILL
- Bug fix in ROQ likelihood
- Distance and phase marginalisation work with ROQ likelihood
- Cpnest now creates checkpoints (resume files) by default

### Removed
-

## [0.4.0] 2019-02-15

### Changed
- Changed the logic around redundancy tests in the `PriorDict` classes
- Fixed an accidental addition of astropy as a first-class dependency and added a check for missing dependencies to the C.I.
- Fixed a bug in the "create-your-own-time-domain-model" example
- Added citation guide to the readme

## [0.3.6] 2019-02-10

### Added
- Added the PolyChord sampler, which can be accessed by using `sampler='pypolychord'` in `run_sampler`
- `emcee` now writes all progress to disk and can resume from a previous run.

### Changed
- Cosmology generalised, users can now specify the cosmology used, default is astropy Planck15
- UniformComovingVolume prior *requires* the name to be one of "luminosity_distance", "comoving_distance", "redshift"
- Time/frequency array generation/conversion improved. We now impose `duration` is an integer multiple of
  `sampling_frequency`. Converting back and forth between time/frequency arrays now works for all valid arrays.
- Updates the bilby.core.utils constants to match those of Astropy v3.0.4
- Improve the load_data_from_cache_file method

### Removed
- Removed deprecated `PriorSet` classes. Use `PriorDict` instead.

## [0.3.5] 2019-01-25

### Added
- Reduced Order Quadrature likelihood
- PTMCMCSampler
- CBC result class
- Additional tutorials on using GraceDB and experts guide on running on events in open data

### Changed
- Updated repository information in Dockerfile for PyMultinest

## [0.3.4] 2019-01-10

### Changes
- Renamed the "basic_tutorial.py" example to "fast_tutorial.py" and created a
 "standard_15d_cbc_tutorial.py"
- Renamed "prior" to "priors" in bilby.gw.likelihood.GravtitationalWaveTransient
  for consistency with bilby.core. **WARNING**: This will break scripts which
  use marginalization.
- Added `outdir` kwarg for plotting methods in `bilby.core.result.Result`. This makes plotting
into custom destinations easier.
- Fixed definition of matched_filter_snr, the interferometer method has become `ifo.inner_product`.

### Added
- log-likelihood evaluations for pymultinest
## [0.3.3] 2018-11-08

Changes currently on master, but not under a tag.

- Removed unnecessary arguments (`ra`, `dec`, `geocent_time`, `psi`) from source functions and replaced them with `**kwargs` where appropriate.
- Renamed `PriorSet` to `PriorDict`
- Renamed `BBHPriorSet` to `BBHPriorDict`
- Renamed `BNSPriorSet` to `BNSPriorDict`
- Renamed `CalibrationPriorSet` to `CalibrationPriorDict`
- Added method to result to get injection recovery credible levels
- Added function to generate a pp-plot from many results to core/result.py
- Fixed a bug which caused `Interferometer.detector_tensor` not to update when `latitude`, `longitude`, `xarm_azimuth`, `yarm_azimuth`, `xarm_tilt`, `yarm_tilt` were updated.
- Added implementation of the ROQ likelihood. The basis needs to be specified by the user.
- Extracted time and frequency series behaviour from `WaveformGenerator` and `InterferometerStrainData` and moved it to `series.gw.CoupledTimeAndFrequencySeries`

### Changes
- Switch the ordering the key-word arguments in `result.read_in_result` to put
  `filename` first. This allows users to quickly read in results by filename
- Result object no longer a child of `dict`. Additionally, the list of
  attributes and saved attributes is standardised
- The above changes effect the saving of posteriors. Users can expect that
  opening files made in python 2(3) which where written in 3(2) may no longer
  work. It was felt that the overheads of maintaining cross-version
  compatibility were too much. Note, working in only python 2 or 3, we do not
  expect users to encounter issues.
- Intermediate data products of samples, nested_samples are stored in the h5
- Time marginalised GravitationalWaveTransient works with arbitrary time priors.

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
- Adds documentation on setting data (https://lscsoft.docs.ligo.org/bilby/transient-gw-data.html)
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
- A `PriorSet` to handle sets of priors and allows reading in from a standardised prior file (see https://lscsoft.docs.ligo.org/bilby/prior.html).
- A standardised file for storing detector data.

### Removed
- All chainconsumer dependency as this was causing issues.
