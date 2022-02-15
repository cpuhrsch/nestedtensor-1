__version__ = '0.1.4+1fc5c63'
git_version = '1fc5c630cea792f5d2fb24fbb01b59bfb3f4f27f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
