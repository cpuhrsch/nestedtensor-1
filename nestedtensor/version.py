__version__ = '0.0.1.dev202012320+a6bd82d'
git_version = 'a6bd82d2673cc0cb5ccfa18a2ba55882834da1d7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
