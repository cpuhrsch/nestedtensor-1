__version__ = '0.1.4+e0f0968'
git_version = 'e0f0968933d8b38ff4cf259fcb9e6f779378de4a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
