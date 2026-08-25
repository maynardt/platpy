"""
platpy — Pipeline for Layered Analysis of Transcriptomics

A Python framework for spatial transcriptomics analysis providing a unified
micron-space coordinate system across imaging, spot, and transcript data.
"""

from .io_utils import load_sample, save_sample
from .viewport import ViewPort
from .dataport import DataPort
from .channels import SampleSet, ImageChannel, OmicChannel, CellChannel, ROIChannel
from .transform_utils import SpatialMetadata

__version__ = "0.1.0"
__all__ = [
    "load_sample", "save_sample",
    "ViewPort", "DataPort",
    "SampleSet", "ImageChannel", "OmicChannel", "CellChannel", "ROIChannel",
    "SpatialMetadata",
]
