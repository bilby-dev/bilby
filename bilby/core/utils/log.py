import json
import logging
from pathlib import Path
import subprocess
import sys
from importlib import metadata

logger = logging.getLogger('bilby')


def setup_logger(outdir='.', label=None, log_level='INFO', print_version=False):
    """ Setup logging output: call at the start of the script to use

    Parameters
    ==========
    outdir, label: str
        If supplied, write the logging output to outdir/label.log
    log_level: str, optional
        ['debug', 'info', 'warning']
        Either a string from the list above, or an integer as specified
        in https://docs.python.org/2/library/logging.html#logging-levels
    print_version: bool
        If true, print version information
    """

    if isinstance(log_level, str):
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            raise ValueError('log_level {} not understood'.format(log_level))
    else:
        level = int(log_level)

    logger = logging.getLogger('bilby')
    logger.propagate = False
    logger.setLevel(level)

    if not any([isinstance(h, logging.StreamHandler) for h in logger.handlers]):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(name)s %(levelname)-8s: %(message)s', datefmt='%H:%M'))
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if not any([isinstance(h, logging.FileHandler) for h in logger.handlers]):
        if label:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            log_file = '{}/{}.log'.format(outdir, label)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)-8s: %(message)s', datefmt='%H:%M'))

            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    if print_version:
        version = get_version_information()
        logger.info('Running bilby version: {}'.format(version))


def get_version_information():
    from bilby import __version__
    return __version__


def loaded_modules_dict():
    module_names = list(sys.modules.keys())
    vdict = {}
    for key in module_names:
        if "." not in key:
            try:
                vdict[key] = metadata.version(key)
            except metadata.PackageNotFoundError:
                continue
    return vdict


def env_package_list(as_dataframe=False):
    """Get the list of packages installed in the system prefix.

    If it is detected that the system prefix is part of a Conda environment,
    a call to ``conda list --prefix {sys.prefix}`` will be made, otherwise
    the call will be to ``{sys.executable} -m pip list installed``.

    Parameters
    ----------
    as_dataframe: bool
        return output as a `pandas.DataFrame`

    Returns
    -------
    pkgs : `list` of `dict`, or `pandas.DataFrame`
        If ``as_dataframe=False`` is given, the output is a `list` of `dict`,
        one for each package, at least with ``'name'`` and ``'version'`` keys
        (more if `conda` is used).
        If ``as_dataframe=True`` is given, the output is a `DataFrame`
        created from the `list` of `dicts`.
    """
    prefix = sys.prefix

    # if a conda-meta directory exists, this is a conda environment, so
    # use conda to print the package list
    conda_detected = (Path(prefix) / "conda-meta").is_dir()
    if conda_detected:
        try:
            pkgs = json.loads(subprocess.check_output([
                "conda",
                "list",
                "--prefix", prefix,
                "--json"
            ]))
        except (FileNotFoundError, subprocess.CalledProcessError):
            # When a conda env is in use but conda is unavailable
            conda_detected = False

    # otherwise try and use Pip
    if not conda_detected:
        try:
            import pip  # noqa: F401
        except ModuleNotFoundError:  # no pip?
            # not a conda environment, and no pip, so just return
            # the list of loaded modules
            modules = loaded_modules_dict()
            pkgs = [{"name": x, "version": y} for x, y in modules.items()]
        else:
            pkgs = json.loads(subprocess.check_output([
                sys.executable,
                "-m", "pip",
                "list", "installed",
                "--format", "json",
            ]))

    # convert to recarray for storage
    if as_dataframe:
        from pandas import DataFrame
        return DataFrame(pkgs)
    return pkgs
