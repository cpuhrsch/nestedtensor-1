__version__ = '0.1.4+d2b7713'
git_version = 'd2b7713e4d9e91151e56d9f615233d350a1c6899'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
