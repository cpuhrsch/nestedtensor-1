__version__ = '0.0.1.dev2020362+c62e485'
git_version = 'c62e485a75d1d6a31d6444a940929862b1fb3a49'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
