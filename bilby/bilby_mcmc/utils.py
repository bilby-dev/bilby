from collections import namedtuple

LOGLKEY = "logl"
LOGLLATEXKEY = r"$\log\mathcal{L}$"
LOGPKEY = "logp"
LOGPLATEXKEY = r"$\log\pi$"

ConvergenceInputs = namedtuple(
    "ConvergenceInputs",
    [
        "autocorr_c",
        "burn_in_nact",
        "thin_by_nact",
        "fixed_discard",
        "target_nsamples",
        "stop_after_convergence",
        "L1steps",
        "L2steps",
        "min_tau",
        "fixed_tau",
        "tau_window",
    ],
)

ParallelTemperingInputs = namedtuple(
    "ParallelTemperingInputs",
    [
        "ntemps",
        "nensemble",
        "Tmax",
        "Tmax_from_SNR",
        "initial_betas",
        "adapt",
        "adapt_t0",
        "adapt_nu",
        "pt_ensemble",
    ],
)
