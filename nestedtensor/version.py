__version__ = '0.0.1.dev202082722+a8160e6'
git_version = 'a8160e60e2e8ff1202ef5d395e8d08322259fa7a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
