__version__ = '0.0.1.dev2020362+15f14ad'
git_version = '15f14adf0fb2fc403f2f09673dbb5224a0f244f8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
