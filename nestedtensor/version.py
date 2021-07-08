__version__ = '0.1.4+b57679c'
git_version = 'b57679c8f2a4fd44f4fbbc50641b1a3617c3d0b5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
