# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

"""Fast image histogram computation.

This module provides fast, multi-threaded histogram computation for image data.
It supports uint8 and uint16 data types, arbitrary bit depths, multi-component
images, optional per-pixel masking, and histogram accumulation.
"""

from ihist._ihist_bindings import histogram

__all__ = [
    "histogram",
]
