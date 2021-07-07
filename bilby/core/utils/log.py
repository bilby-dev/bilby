import logging
import os
from pathlib import Path
import sys

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

    if type(log_level) is str:
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            raise ValueError('log_level {} not understood'.format(log_level))
    else:
        level = int(log_level)

    logger = logging.getLogger('bilby')
    logger.propagate = False
    logger.setLevel(level)

    if any([type(h) == logging.StreamHandler for h in logger.handlers]) is False:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(name)s %(levelname)-8s: %(message)s', datefmt='%H:%M'))
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if any([type(h) == logging.FileHandler for h in logger.handlers]) is False:
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
    version_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.version')
    try:
        with open(version_file, 'r') as f:
            return f.readline().rstrip()
    except EnvironmentError:
        print("No version information file '.version' found")


def loaded_modules_dict():
    module_names = list(sys.modules.keys())
    vdict = {}
    for key in module_names:
        if "." not in key:
            vdict[key] = str(getattr(sys.modules[key], "__version__", "N/A"))
    return vdict
