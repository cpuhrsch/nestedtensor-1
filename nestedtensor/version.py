__version__ = '0.0.1.dev202011220+add1676'
git_version = 'add167623edaa070509633378fdb1cbe8791c853'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
