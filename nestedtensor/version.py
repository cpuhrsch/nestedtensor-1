__version__ = '0.1.4+3c761e7'
git_version = '3c761e7e7ca57fd779cc0c79fc63b016238ca3af'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
