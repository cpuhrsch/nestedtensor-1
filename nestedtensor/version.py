__version__ = '0.0.1.dev20207117+5731726'
git_version = '573172687111bee6d3de91a9615d71a09b163e48'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
