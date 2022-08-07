import argparse
import logging
import os
import subprocess

from .log import logger


def set_up_command_line_arguments():
    """ Sets up command line arguments that can be used to modify how scripts are run.

    Returns
    =======
    command_line_args, command_line_parser: tuple
        The command_line_args is a Namespace of the command line arguments while
        the command_line_parser can be given to a new `argparse.ArgumentParser`
        as a parent object from which to inherit.

    Notes
    =====
        The command line arguments are passed initially at runtime, but this parser
        does not have a `--help` option (i.e., the command line options are
        available for any script which includes `import bilby`, but no help command
        is available. This is done to avoid conflicts with child argparse routines
        (see the example below).

    Examples
    ========
    In the following example we demonstrate how to setup a custom command line for a
    project which uses bilby.

    .. code-block:: python

        # Here we import bilby, which initialises and parses the default command-line args
        >>> import bilby
        # The command line arguments can then be accessed via
        >>> bilby.core.utils.command_line_args
        Namespace(clean=False, log_level=20, quite=False)
        # Next, we import argparse and define a new argparse object
        >>> import argparse
        >>> parser = argparse.ArgumentParser(parents=[bilby.core.utils.command_line_parser])
        >>> parser.add_argument('--argument', type=int, default=1)
        >>> args = parser.parse_args()
        Namespace(clean=False, log_level=20, quite=False, argument=1)

    Placing these lines into a script, you'll be able to pass in the usual bilby default
    arguments, in addition to `--argument`. To see a list of all options, call the script
    with `--help`.

    """
    try:
        parser = argparse.ArgumentParser(
            description="Command line interface for bilby scripts",
            add_help=False, allow_abbrev=False)
    except TypeError:
        parser = argparse.ArgumentParser(
            description="Command line interface for bilby scripts",
            add_help=False)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help=("Increase output verbosity [logging.DEBUG]." +
                              " Overridden by script level settings"))
    parser.add_argument("-q", "--quiet", action="store_true",
                        help=("Decrease output verbosity [logging.WARNING]." +
                              " Overridden by script level settings"))
    parser.add_argument("-c", "--clean", action="store_true",
                        help="Force clean data, never use cached data")
    parser.add_argument("-u", "--use-cached", action="store_true",
                        help="Force cached data and do not check its validity")
    parser.add_argument("--sampler-help", nargs='?', default=False,
                        const='None', help="Print help for given sampler")
    parser.add_argument("--bilby-test-mode", action="store_true",
                        help=("Used for testing only: don't run full PE, but"
                              " just check nothing breaks"))
    parser.add_argument("--bilby-zero-likelihood-mode", action="store_true",
                        help=("Used for testing only: don't run full PE, but"
                              " just check nothing breaks"))
    args, unknown_args = parser.parse_known_args()
    if args.quiet:
        args.log_level = logging.WARNING
    elif args.verbose:
        args.log_level = logging.DEBUG
    else:
        args.log_level = logging.INFO

    return args, parser


def run_commandline(cl, log_level=20, raise_error=True, return_output=True):
    """Run a string cmd as a subprocess, check for errors and return output.

    Parameters
    ==========
    cl: str
        Command to run
    log_level: int
        See https://docs.python.org/2/library/logging.html#logging-levels,
        default is '20' (INFO)

    """

    logger.log(log_level, 'Now executing: ' + cl)
    if return_output:
        try:
            out = subprocess.check_output(
                cl, stderr=subprocess.STDOUT, shell=True,
                universal_newlines=True)
        except subprocess.CalledProcessError as e:
            logger.log(log_level, 'Execution failed: {}'.format(e.output))
            if raise_error:
                raise
            else:
                out = 0
        os.system('\n')
        return out
    else:
        process = subprocess.Popen(cl, shell=True)
        process.communicate()
