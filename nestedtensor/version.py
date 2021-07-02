__version__ = '0.1.4+dcb244d'
git_version = 'dcb244d36115512ffb30344f5d01f3e2bc9ae027'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
