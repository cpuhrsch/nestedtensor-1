__version__ = '0.0.1.dev20204122+21c1749'
git_version = '21c17492d53f2d55a523db007f6438a2b1d1ed7a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
