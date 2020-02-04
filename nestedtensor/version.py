__version__ = '0.0.1.dev2020245+3ccc876'
git_version = '3ccc87655da02f33ccf3c0009ec7ffa83daaf448'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
