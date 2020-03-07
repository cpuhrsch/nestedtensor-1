__version__ = '0.0.1.dev2020375+8c1d0f5'
git_version = '8c1d0f567ec115032ae7340c5e53f69404fca39e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
