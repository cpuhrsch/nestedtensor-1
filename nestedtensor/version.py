__version__ = '0.1.4+1f18ce0'
git_version = '1f18ce0faec6b3b08c3a46ebb40551bb1cb6b679'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
