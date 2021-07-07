__version__ = '0.1.4+182d8d2'
git_version = '182d8d215a3b3ef456a0b334c82201e95c8b095a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
