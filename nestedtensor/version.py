__version__ = '0.0.1.dev202082720+9ddd5b4'
git_version = '9ddd5b45c874e6c197b911619647cf1ceeb33ef4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
