__version__ = '0.0.1.dev202081218+0d6093b'
git_version = '0d6093bb4d7f8c68b60978e6e736e2d19654abff'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
