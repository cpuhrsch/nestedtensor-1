__version__ = '0.0.1.dev202082722+f5f2d7a'
git_version = 'f5f2d7a406c79445f04b55ce33301405d56b163a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
