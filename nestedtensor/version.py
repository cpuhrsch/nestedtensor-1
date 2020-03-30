__version__ = '0.0.1.dev20203308+550b5e2'
git_version = '550b5e295103cde73765c1cfa02e4b4231548c03'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
