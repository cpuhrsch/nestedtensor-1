__version__ = '0.1.4+95cd13e'
git_version = '95cd13e26a05622bcde692136d2f48b858eef9d0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
