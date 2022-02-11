__version__ = '0.1.4+4b4db45'
git_version = '4b4db451c825d37c4ed5639a321783c82b4f9c98'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
