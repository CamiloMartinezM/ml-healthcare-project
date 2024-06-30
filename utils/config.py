# -*- coding: utf-8 -*-
#
# File: utils/config.py
# Description: This file defines the configuration settings for the project.

import logging
import multiprocessing
from warnings import warn

CPU_COUNT = multiprocessing.cpu_count()

try:
    import cuml

    CUML_INSTALLED = True
except ImportError:
    warn(
        "cuML is not installed, so we cannot use GPU-acceleration. Using sklearn's version instead."
    )
    CUML_INSTALLED = False

try:
    import skopt

    SKOPT_INSTALLED = True
except ImportError:
    warn(
        "scikit-optimize is not installed, so we cannot use Bayesian optimization. Using sklearn's GridSearchCV instead."
    )
    SKOPT_INSTALLED = False

# Suppress Numba CUDA driver debug and info messages
logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)

# Set the logging level for matplotlib to WARNING to suppress debug messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)
