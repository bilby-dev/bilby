import multiprocessing
from contextlib import contextmanager

from .log import logger


def create_pool(
    likelihood,
    priors,
    use_ratio=None,
    search_parameter_keys=None,
    npool=None,
    pool=None,
    parameters=None,
):
    from ...core.sampler.base_sampler import _initialize_global_variables

    if parameters is None:
        parameters = dict()

    _pool = None
    if pool == "mpi":
        try:
            from schwimmbad import MPIPool
        except ImportError:
            raise ImportError("schwimmbad must be installed to use MPI pool")

        _initialize_global_variables(
            likelihood=likelihood,
            priors=priors,
            search_parameter_keys=search_parameter_keys,
            use_ratio=use_ratio,
            parameters=parameters,
        )
        _pool = MPIPool(use_dill=True)
        if _pool.is_master():
            logger.info(f"Created MPI pool with size {_pool.size}")
    elif pool is not None:
        _pool = pool
    elif npool is not None:
        _pool = multiprocessing.Pool(
            processes=npool,
            initializer=_initialize_global_variables,
            initargs=(likelihood, priors, search_parameter_keys, use_ratio, parameters),
        )
        logger.info(f"Created multiprocessing pool with size {npool}")
    else:
        _pool = None
    return _pool


def close_pool(pool):
    if hasattr(pool, "close"):
        pool.close()
    else:
        import IPython; IPython.embed()
    if hasattr(pool, "join"):
        pool.join()


@contextmanager
def bilby_pool(
    likelihood, priors,
    use_ratio=None,
    search_parameter_keys=None,
    npool=None,
    pool=None,
    parameters=None,
):
    if hasattr(pool, "map"):
        user_pool = True
    else:
        user_pool = False

    try:
        _pool = create_pool(
            likelihood=likelihood,
            priors=priors,
            search_parameter_keys=search_parameter_keys,
            use_ratio=use_ratio,
            npool=npool,
            pool=pool,
            parameters=parameters,
        )
        yield _pool
    finally:
        if not user_pool and "_pool" in locals():
            close_pool(_pool)
