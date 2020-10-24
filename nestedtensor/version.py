__version__ = '0.0.1.dev2020102322+66f6323'
git_version = '66f63236a18e4f05ef545ce25cff73339ffc09ff'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
