# SPDX-License-Identifier: Apache-2.0
from .model import Detector
from .hub import load_detector, predict_image

__all__ = ['Detector', 'load_detector', 'predict_image']
__version__ = '1.0.0'
