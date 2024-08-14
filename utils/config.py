# -*- coding: utf-8 -*-
#
# File: utils/config.py
# Description: This file defines the configuration settings for the project.

import logging
import multiprocessing
import shutil
from importlib.util import find_spec as importlib_find_spec
from os.path import join as path_join
from warnings import simplefilter as warnings_simplefilter
from warnings import warn

import matplotlib.pyplot as plt

# Paths
CACHE_DIR = path_join(".", ".cache")  # Cache directory for storing intermediate results
DATA_DIR = path_join(".", "data")  # Data directory where raw data is stored
PLOTS_DIR = path_join(".", "results", "plots")  # Directory to save plots

if not shutil.which("latex"):
    warn("LaTeX is not installed. Some plots may not render correctly.")
    LATEX_INSTALLED = False
else:
    LATEX_INSTALLED = True

SCIENCEPLOTS_INSTALLED = importlib_find_spec("scienceplots") is not None

if SCIENCEPLOTS_INSTALLED:
    import scienceplots

    PAPER_STYLE = ["science", "ieee"]
    PAPER_STYLE += ["no-latex"] # if not LATEX_INSTALLED else []
    NOTEBOOK_STYLE = ["notebook"]
else:
    warn(
        "The scienceplots package is not installed. Please install it for better plot styles."
        + "See: https://github.com/garrettj403/SciencePlots"
    )
    PAPER_STYLE = "fivethirtyeight"
    NOTEBOOK_STYLE = "fivethirtyeight"

CPU_COUNT = multiprocessing.cpu_count()

CUML_INSTALLED = importlib_find_spec("cuml") is not None

if not CUML_INSTALLED:
    warn(
        "cuML is not installed, so we cannot use GPU-acceleration. Using sklearn's version instead."
    )

SKOPT_INSTALLED = importlib_find_spec("skopt") is not None

if not SKOPT_INSTALLED:
    warn("scikit-optimize is not installed. Using sklearn's GridSearchCV instead.")

OPENTSNE_INSTALLED = importlib_find_spec("openTSNE") is not None

# Prefer using opentsne instead of sklearn.manifold.TSNE because it is faster by using the FFT method
# Install with: conda install --channel conda-forge opentsne
# See: https://github.com/pavlin-policar/openTSNE/
if not OPENTSNE_INSTALLED:
    warn("opentsne is not available. Using sklearn.manifold.TSNE instead.")

TSNE_CUDA_INSTALLED = importlib_find_spec("tsnecuda") is not None

if TSNE_CUDA_INSTALLED:
    # If `use_tsnecuda` is specified when calling `apply_tsne`, then we use the CUDA version of t-SNE
    # See: https://github.com/CannyLab/tsne-cuda/blob/main/INSTALL.md
    # Install with: conda install tsnecuda -c conda-forge
    warn("tsnecuda library is not available. Using opentsne instead.")

CUPY_INSTALLED = importlib_find_spec("cupy") is not None

if not CUPY_INSTALLED:
    warn("CuPy is not installed. Using NumPy instead.")

# Suppress Numba CUDA driver debug and info messages
logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)

# Set the logging level for matplotlib to WARNING to suppress debug messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Ignore repeated instances of the same warning
warnings_simplefilter("ignore", RuntimeWarning)

# DPI for plots (change when final plotting)
DPI = 200
plt.rcParams["figure.dpi"] = DPI
