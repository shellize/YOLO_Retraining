from __future__ import annotations

import multiprocessing


SAFE_START_METHOD = "spawn"


def configure_safe_start_method() -> str:
    """Use a CUDA-safe start method for every YOLOv5 DataLoader worker.

    Linux defaults to ``fork``, but these entry points initialize CUDA before
    YOLOv5 constructs its train and validation DataLoaders.  Forking after that
    initialization can inherit locks whose owning threads do not exist in the
    child process.  ``spawn`` starts each worker in a clean interpreter instead.
    """
    if multiprocessing.get_start_method(allow_none=True) != SAFE_START_METHOD:
        multiprocessing.set_start_method(SAFE_START_METHOD, force=True)
    return multiprocessing.get_start_method()
