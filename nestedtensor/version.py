__version__ = '0.0.1.dev20202120+ef02296'
git_version = 'ef02296154e15ce1105ff0e04051aae8602b1874'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
