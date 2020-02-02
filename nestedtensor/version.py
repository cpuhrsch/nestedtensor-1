__version__ = '0.0.1.dev2020221+05c8e56'
git_version = '05c8e56fc98a4842b1f4d446d72cba640610e16b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
