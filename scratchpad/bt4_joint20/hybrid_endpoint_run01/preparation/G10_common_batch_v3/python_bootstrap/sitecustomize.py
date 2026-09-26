"""This fixed CPU operation also caps fresh multiprocessing-spawn interpreters."""
import os

try:
    from numcodecs import blosc
    blosc.set_nthreads(2)
    if blosc.get_nthreads() != 2:
        raise RuntimeError('Blosc thread cap did not take effect')
except BaseException:
    # Python normally continues after sitecustomize errors; this operation must not.
    os._exit(78)
