__version__ = '0.0.1.dev202091218+52eecaf'
git_version = '52eecaf198ab056b6aeea08f6da7271e9cd4a68c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
