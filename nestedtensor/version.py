__version__ = '0.1.4+f7b58d5'
git_version = 'f7b58d5daef85db2d0498dca67c7a9c287b52251'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
