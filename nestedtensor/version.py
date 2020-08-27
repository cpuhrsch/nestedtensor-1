__version__ = '0.0.1.dev202082722+0b8c3c0'
git_version = '0b8c3c0de10a9b04aaf41763faef7cafaf87fcef'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
