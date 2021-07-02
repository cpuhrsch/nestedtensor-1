__version__ = '0.1.4+1a97fd9'
git_version = '1a97fd9537aabf9a73ed1bacfe2f3e8fc6e4a161'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
