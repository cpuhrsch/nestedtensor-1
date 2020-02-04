__version__ = '0.0.1.dev2020244+8870bb6'
git_version = '8870bb64a2cacc5845dc8c84b8fb2f1cc745f16c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
