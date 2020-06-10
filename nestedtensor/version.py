__version__ = '0.0.1.dev202061022+b2df5ed'
git_version = 'b2df5ed101b83dfd560aea77f0f79c56a9940e53'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
