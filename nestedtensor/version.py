__version__ = '0.0.1.dev20207118+aba2b40'
git_version = 'aba2b40722890875ac389269a96942f03e238a4b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
