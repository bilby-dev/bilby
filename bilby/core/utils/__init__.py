from . import random
from .calculus import *
from .cmd import *
from .colors import *
from .constants import *
from .conversion import *
from .counter import *
from .docs import *
from .entry_points import *
from .env import *
from .introspection import *
from .io import *
from .log import *
from .meta_data import *
from .plotting import *
from .samples import *
from .series import *

#  Instantiate the default argument parser at runtime
command_line_args, command_line_parser = set_up_command_line_arguments()

# As of bilby 2.8, we follow the Python logging recommendation for
# libraries and do NOT install a StreamHandler at import time. The
# ``bilby`` logger has only a NullHandler attached (see log.py). This
# avoids interfering with downstream applications that want to manage
# their own logging configuration.
#
# If the user runs bilby with a ``--log-level`` command-line argument we
# honour it by calling :func:`setup_logger` explicitly, matching the
# pre-2.8 behaviour for scripts that rely on this pattern.
if command_line_args.log_level != 'INFO':
    setup_logger(print_version=False, log_level=command_line_args.log_level)
