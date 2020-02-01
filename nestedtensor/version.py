__version__ = '0.0.1.dev2020211+ef9ca4a'
git_version = 'ef9ca4ab5d9a153b1cf1f4a841c410c0f6476d27'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
