"""
platpy — Pipeline for Layered Analysis of Transcriptomics

A Python framework for spatial transcriptomics analysis providing a unified
micron-space coordinate system across imaging, spot, and transcript data.
"""

from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNotFoundError

from .io_utils import load_sample, save_sample
from .viewport import ViewPort
from .dataport import DataPort
from .channels import SampleSet, ImageChannel, OmicChannel, CellChannel, ROIChannel
from .transform_utils import SpatialMetadata

try:
    # Single source of truth is pyproject.toml's [project] version, read from
    # installed package metadata — avoids a second hardcoded string drifting
    # out of sync with what's actually published (caught 2026-09: __version__
    # here still said 0.1.0 after the real PyPI release had moved to 0.1.1).
    __version__ = _pkg_version("platpy")
except _PkgNotFoundError:
    # Not installed (e.g. running from a source checkout with no pip install)
    __version__ = "0.0.0+unknown"
__all__ = [
    "load_sample", "save_sample",
    "ViewPort", "DataPort",
    "SampleSet", "ImageChannel", "OmicChannel", "CellChannel", "ROIChannel",
    "SpatialMetadata",
]
