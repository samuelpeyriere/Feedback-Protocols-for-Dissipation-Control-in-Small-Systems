"""Random numbers of the simulations, reproducible across joblib workers.

The simulators draw their random numbers through `random` and `normal`. Without a call to
`seed`, these come from numpy's global RNG (unseeded). `seed(s)` gives the calling thread its
own stream seeded with `s`, and `seeded(jobs)` gives each joblib job its own seed, drawn from
the caller's stream before the jobs are dispatched. The results then depend neither on which
worker (process or thread) runs which job, nor on the number of workers.

    hp.seed(0)
    results = Parallel(n_jobs=-1)(hp.seeded(delayed(f)(x) for x in xs))
"""
import threading

import numpy as np

_local = threading.local()


def _stream():
    return getattr(_local, 'state', np.random)


def seed(s) -> None:
    """Seed the random stream of the calling thread (`s`: int or sequence of ints)."""
    _local.state = np.random.RandomState(s)


def random(size=None):
    return _stream().random_sample(size)


def normal(loc=0.0, scale=1.0, size=None):
    return _stream().normal(loc, scale, size)


def _seeded_call(s, func, args, kwargs):
    # restore the caller's stream afterwards, in case the job runs in the calling thread
    previous = getattr(_local, 'state', None)
    seed(s)
    try:
        return func(*args, **kwargs)
    finally:
        if previous is None:
            del _local.state
        else:
            _local.state = previous


def seeded(jobs) -> list:
    """The joblib jobs `jobs` (from `delayed`), each made to run with its own seed."""
    jobs = list(jobs)
    seeds = _stream().randint(2**32, size=len(jobs), dtype=np.int64)
    return [(_seeded_call, (int(s), func, args, kwargs), {}) for s, (func, args, kwargs) in zip(seeds, jobs)]
