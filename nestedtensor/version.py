__version__ = '0.1.4+ed972fc'
git_version = 'ed972fc24ef8f32801803d0ec1d4f2a8f854d0b9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
