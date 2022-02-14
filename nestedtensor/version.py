__version__ = '0.1.4+8df8402'
git_version = '8df84026e4027ed768015bda351213336ba68c5d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
