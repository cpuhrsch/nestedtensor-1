__version__ = '0.0.1.dev2020210+a5ed5cc'
git_version = 'a5ed5ccf2462e9bdb732d0bb0f59a361ae3c79ed'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
