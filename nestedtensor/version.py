__version__ = '0.0.1.dev2020246+a49da07'
git_version = 'a49da07c50c52492f5979e517c8794e57a716057'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
