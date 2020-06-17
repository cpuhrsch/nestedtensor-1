__version__ = '0.0.1.dev202061718+a924abd'
git_version = 'a924abd75cbcfa1bf13639e5f9f37f81c2dbe14e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
