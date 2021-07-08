__version__ = '0.1.4+6460c68'
git_version = '6460c681907d64081b49abab4a6487d590e54780'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
