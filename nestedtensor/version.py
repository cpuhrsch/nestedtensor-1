__version__ = '0.1.4+3300e3b'
git_version = '3300e3bc42394ab4bb226cef8acc631012a72ef0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
