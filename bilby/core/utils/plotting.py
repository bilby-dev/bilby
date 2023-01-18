import functools
import os
import shutil

from .log import logger


def latex_plot_format(func):
    """
    Wrap the plotting function to set rcParams dependent on environment variables

    The rcparams can be set directly from the env. variable `BILBY_STYLE` to
    point to a matplotlib style file. Or, if `BILBY_STYLE=default` (any case) a
    default setup is used, this is enabled by default. To not use any rcParams,
    set `BILBY_STYLE=none`. Occasionally, issues arrise with the latex
    `mathdefault` command. A fix is to define this command in the rcParams. An
    env. variable `BILBY_MATHDEFAULT` can be used to turn this fix on/off.
    Setting `BILBY_MATHDEFAULT=1` will enable the fix, all other choices
    (including undefined) will disable it. Additionally, the BILBY_STYLE and
    BILBY_MATHDEFAULT arguments can be passed into any
    latex_plot_format-wrapped plotting function and will be set directly.

    """
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        if "BILBY_STYLE" in kwargs:
            bilby_style = kwargs.pop("BILBY_STYLE")
        else:
            bilby_style = os.environ.get("BILBY_STYLE", "default")

        if "BILBY_MATHDEFAULT" in kwargs:
            bilby_mathdefault = kwargs.pop("BILBY_MATHDEFAULT")
        else:
            bilby_mathdefault = int(os.environ.get("BILBY_MATHDEFAULT", "0"))

        if bilby_mathdefault == 1:
            logger.debug("Setting mathdefault in the rcParams")
            rcParams['text.latex.preamble'] = r'\providecommand{\mathdefault}[1][]{}'

        logger.debug("Using BILBY_STYLE={}".format(bilby_style))
        if bilby_style.lower() == "none":
            return func(*args, **kwargs)
        elif os.path.isfile(bilby_style):
            plt.style.use(bilby_style)
            return func(*args, **kwargs)
        elif bilby_style in plt.style.available:
            plt.style.use(bilby_style)
            return func(*args, **kwargs)
        elif bilby_style.lower() == "default":
            _old_tex = rcParams["text.usetex"]
            _old_serif = rcParams["font.serif"]
            _old_family = rcParams["font.family"]
            if shutil.which("latex"):
                rcParams["text.usetex"] = True
            else:
                rcParams["text.usetex"] = False
            rcParams["font.serif"] = "Computer Modern Roman"
            rcParams["font.family"] = "serif"
            rcParams["text.usetex"] = _old_tex
            rcParams["font.serif"] = _old_serif
            rcParams["font.family"] = _old_family
            return func(*args, **kwargs)
        else:
            logger.debug(
                "Environment variable BILBY_STYLE={} not used"
                .format(bilby_style)
            )
            return func(*args, **kwargs)
    return wrapper_decorator
