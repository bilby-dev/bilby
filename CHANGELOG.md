# All notable changes will be documented in this file

## [1.0.1] 2020-08-29

Version 1.0.1 release of bilby

### Added
- Added an rcparams configuration for plotting (!832)
- Added `chi_1` and `chi_2` parameters to default latex label dictionary (!841)
- Allow output merged result file to be gzip or saved as a HDF5 file (!802)

### Changes
- Fixed first value in EOS cumulative integral(!860)
- Fixed saving the number of likelihood evaluations (!848)
- Likelihood condition is now strictly increasing (!846)
- Fixed a minor issue with conditional priors that could cause unexpected behaviour in edge cases (!838)
- Fixed `__repr__` method in the `FromFile` prior (!836)
- Fixed an issue that caused problems for some users when plotting with a latex backend (!816)
- Fixed bug that occured when min/max of interpolated priors was changed (!815)
- Fixed time domain waveform epoch (!736)
- Fixed time keeping in multinest (!830)
- Now checks if marginalised priors were defined before marginalising (!829)
- Fixed an issue with multivariate Gaussian prior (!822)
- Various minor code improvements (!836)(!839)
- Various minor bug fixes and improvements to the documentation (!820)(!823)(!837)
- Various testing improvements (!833)(!847)(!855)(!852)

## [1.0.0] 2020-07-06

Version 1.0 release of bilby

### Changes
- Minor bug fixes and typo changes only from 0.6.9, see
git.ligo.org/lscsoft/bilby/-/merge_requests?scope=all&utf8=%E2%9C%93&state=merged&milestone_title=1.0.0
for details

## [0.6.9] 2020-05-21
### Changes
- Improvement to the proposal step in dynesty (!774)
- Fix a bug in checking and making directories (!792)
- Clean up of the default prior files (!789)

## [0.6.8] 2020-05-13
### Added
- Option to sample in the sky frame (!786)
- Multiprocessing to reconstruction of marginalized parameters (!782)
- Generic reweighting method for likelihood / priors (!776)
- Parameterized EOS sampling (!543)
- Implementation of the UltraNest sampler (!766)
- Implementation of arVix result files (!772)
- Added basic pre-commit behaviour (!763)

### Changes
- Updated the default PSD to O4 (!757)
- Make multinest allow long file names, optional and work with MPI (!764 !785)
- Add min/max to aligned spin prior (!787)
- Reduce redudant code (!703)
- Added testing for python 3.8 (!762)
- Improvements to the waveform plot (!769)

## [0.6.7] 2020-04-15
### Changes
- Allow dynesty to run with multiprocessing (!754)
- Rewrite ptemcee implementation (!750)
- Change 'source frame' to 'detector frame' in L34-35 of compare_samplers tutorial (!745)
- Allow lal dictionary to be passed through to '_base_lal_cbc_fd_waveform' (!752)

## [0.6.6] 2020-03-06
### Changes
- Fix bug where injected values are not present for corner plot (!749)
- Significant backwards-incompatible improvements to `dynesty` checkpointing (!746)
- Improve checkpoint interval calculation with `dynesty` (!741)
- Fix reading of `PriorDict` class from result file (!739)
- Fix definition of time for time-domain `lalsimulation` waveforms (!736)
- LaTeX text formatting for plots by default (!702)

### Added
- Normalisation dynamically computed when using prior constraints (!704)

## [0.6.5] 2020-02-14
### Changes
- Fix for time reconstruction bug (!714)
- Resolved errors Waveform longer than the frequency array (!710)
- Prior reading clean-up (!715)
- More efficient dynesty restarting (!713)
- PP tests show 123 sigma bounds by default (!726)

### Added
- HealPixPrior (!651)
- GW prior documentation (!720)
- Multiple contours to PP tests plots (!721) 
- Distance marginalization for non-luminosity-distance parameters (!719)

### Removed
- Pipenv (!724)


## [0.6.4] 2020-01-30
### Changes
- Discontinue python2.7 support (!697)
- Minor adjustments to the act calculation method (!679, !707)
- Restructure of the prior module (!688)
- Improvements to the documentation (!708, !700)
- Bug fix when maximum < minimum (!696)

### Added
- Improved waveform error handling (!653)
- Waveform check to the CI (!698)

## [0.6.3] 2020-01-03
### Changed
- Fixed an issue with the ROQ segment scaling (!690)

## [0.6.2] 2019-12-20
### Added
- Introduced conditional prior sets (!332)(!673)(!674)
- Introduced joint priors (!668)
- Added a check to make sure sampling time exists before trying to update (!672)
### Changed
- Fixed a caching issue with the waveform generators (!630)
- Fixed an issue that made the dynamic dynesty sampler not work (!667)
- Changed the backend docker files (!669)
- Fixed an error when plotting time domain data when using `filtfilt=True`
- `Interped` priors now dynamically update when new `yy` values are set (!675)
- Fixed the ROQ scaling checks (!678)

## [0.6.1] 2019-12-02

HotFix release following 0.6.0 fixing a minor bug in the generation of derived
parameters.

## [0.6.0] 2019-12-02
### Added
- A bilby-implemenatation of the dynesty rwalk proposal method (!640)
- An ACT estimate to the rwalk option (!643)
- HTML waveform plots and general improvements to the waveform plot (!641, !659)
- Add a function to resample bilby generated with a uniform in mass ratio and
  chirp_mass prior to uniform in component mass (!642)
- Adds checking to the ROQ usage: warning messages generated when out of bounds (!549)
- A safety check to the time reconstrucion (!633)
- Added the kombine sampler (!637)
- Added in-plane spins (!646)
## Changes
- Changed the PriorDict base class to from OrderedDict to just dict. This fixes
  an issue with pickling priors (!652)
- Improvements to the behaviour of the conversion functions (!647)
- Prevent chirp mass railing for the GW150914 examples (!635)


## [0.5.9] 2019-10-25

### Added
- Default reflective boundaries for calibration parameters !623
- Support for inferring method arguments !608

## Changes
- Speed up the prior evaluations by implementing directly with checks to scipy !627
- Soft initalisation option for the Sampler class !620
- Improvements to JSON reading and writing for functions !621
- Fixed bug in prior reading !618 !617
- Fixes to the examples !619 !614 !626 !616
- Update to the test mode storing extra information !613
- Minor improvements to the ptemcee sampler
- Improved contributing guidelines !610

## Removed
- Default printing of bilby version at import !608

## [0.5.8] 2019-09-26

### Added
- Progress bar in post-processing step

### Changed
- Fixed implementation of calibration !607
- Fixed interaction with dynesty for reflective bounds !604
- Fixed behaviour of n_effective with check pointing !603
- Fixed testing of constants !605
- Fixed bug in bilby_result with python2.7

## [0.5.7] 2019-09-19

### Added
- bilby_convert_resume file CL tool for converting dynesty resume files into preresults !599

### Changes
- Change the constants (Msun, REarth etc) to match the values in LAL !597
- Change the Greenwhich Mean Sidereal Time conversion to match the method in LAL !597
- Update dynesty requirement to 1.0.0
- Improve integration of bounds with dynesty !589
- Fixed issue with mutable default argument !596
- Allow the use of n_effective in dynesty !592
- Allow the use of n_periodic in cpnest !591
- Fix bug in dt calc

## [0.5.6] 2019-09-04

### Changes
- Deprecation of the old helper functions (e.g., fetch open data)
- Improvements to the documentation
- Fix a bug in the dt calculations of the GW likelihood
- Various small bug fixes

### Added
- LAL version information in the meta data

## [0.5.5] 2019-08-22

### Added
- Reading/writing of the prior in a JSON format
- Checks for marginalization flags

### Changes
- Improvements to the examples: reorganisation and fixing bugs
- Fixed bug with scipy>=1.3.0 and spline
- Removed the sqrt(2) normalisation from the scalar longitudinal mode
- Improve PSD filename reading (no longer required "/" to read local files)
- Fix bug in emcee chains
- Added a try/except cluase for building the lookup table

## [0.5.4] 2019-07-30

### Added
- Analytic CDFs 
- Reading/writing of grid results objects

### Changed
- Dynesty default settings changed: by default, now uses 30xndim walks. This was
shown (!564) to provide better convergence for the long-duration high-spin tests.
- Fix bug in combined runs log evidence calculations
- Fixed bugs in the nightly tests 

## [0.5.3] 2019-07-23
### Added
- Jitter time marginalization. For the time-marginalized likelihood, a jitter
  is used to ensure proper sampling without artifacts (!534)
- Zero likelihood mode for testing and zero-likelihood test to the nightly C.I ((!542)
- 15D analytic Gaussian test example (!547) 

### Changes
- Dynesty version minimum set to 0.9.7. Changes to this sampler vastly improve
  performance (!537)
- Improvements to waveform plotting (!534) 
- Fixed bugs in the prior loading and added tests (!531 !539 !553 !515)
- Fixed issue in 1D CDF prior plots (!538)
- ROQ weights stored as npz rather than json (memory-performance improvement) (!536)
- Distance marginalisation now uses cubic rather than linear interpolation. Improves
  distance/inclination posteriors for high SNR systems. (!552)
- Inputs to hyperpe modified to allow for more flexible sampling prior specification
  and improve efficiency. (!545)
- Fix definition of some spin phase parameters (!556).

## [0.5.2] 2019-06-18
### Added
- Method to read data in using gwpy get (and associated example)
- Adds a catch for broken resume files with improves reporting

### Changed
- Updated and fixed bugs in examples
- Resolve sampling time persistence for runs which are interupted
- Improvements to the PP plot
- Speed up of the distance calculation
- Fixed a bug in the inteference of bilby command line arguments with user specified command lines
- Generalised the consistency checks for ResultLists
- Fixes to some tests
- Makes the parameter conversion a static method rather than a lambda expression

## [0.5.1] 2019-06-05
### Added
- Option for the GraceDB service URL
- Precessing BNS
- Functionality to make a waveform plot

### Changed
- Changes to ROQ weight generation: finer time-steps and fixed a bug in the time definition
- Fixed typo "CompactBinaryCoalesnce" -> "CompactBinaryCoalescence" (old class now has deprecation warning)
- Fixed a minor bug in the frequency mask caching
- Minor refractoring of the GWT likelihood and detector tests
- Initial samples in dynesty now generated from the constrained prior

## [0.5.0] 2019-05-08

### Added
- A plot_skymap method to the CBCResult object based on ligo.skymap
- A plot_calibration_posterior method to the CBCResult object
- Method to merge results

### Changed
- Significant refactoring of detector module: this should be backward conmpatible. This work was done to break the large detector.py file into smaller, more manageable chunks. 
- The `periodic_boundary` option to the prior classes has been changed to `boundary`.
*This breaks backward compatibility*.
The options to `boundary` are `{'periodic', 'reflective', None}`.
Periodic boundaries are supported as before.
Reflective boundaries are supported in `dynesty` and `cpnest`.  
- Minor speed improvements by caching intermediate steps
- Added state plotting for dynesty. Use `check_point_plot=True` in the `run_sampler` 
function to create trace plots during the dynesty checkpoints
- Dynesty now prints the progress to STDOUT rather than STDERR
- `detector` module refactored into subpackage. Maintains backward compatibility.
- Specifying alternative frequency bounds for the ROQ now possible if the appropriate
`params.dat` file is passed.

### Removed
- Obsolete (and potentially incorrect) plot_skymap methods from gw.utils

## [0.4.5] 2019-04-03

### Added
- Calibration method and plotting
- Multivariate Gaussian prior
- Bayesian model diminsionality calculator
- Dynamic dynesty (note: this is in an alpha stage)
- Waveform caching

### Changes
- Fixed bugs in the ROQ time resolution
- Fixed bugs in the gracedb wrapper-method
- Improvements to the pp-plot method
- Improved checkpointing for emcee/ptemcee
- Various perforance-related improvements

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
