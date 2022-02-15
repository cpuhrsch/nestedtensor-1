__version__ = '0.1.4+4c90281'
git_version = '4c90281009afc6de8aeb74c97d4a342c7109e854'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
