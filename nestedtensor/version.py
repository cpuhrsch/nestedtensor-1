__version__ = '0.0.1.dev202081419+9f227c1'
git_version = '9f227c1a4c949280253db19cb99a1e62aaaf5ca7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
