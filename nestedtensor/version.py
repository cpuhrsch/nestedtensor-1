__version__ = '0.0.1.dev20205303+c9c9643'
git_version = 'c9c9643cd05051c4738956887e563fa7f99f9e0d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
