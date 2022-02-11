__version__ = '0.1.4+5e19db6'
git_version = '5e19db649b9ba719b289eccecfd2b9e55b6d82ca'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
