__version__ = '0.0.1.dev202072720+22e33db'
git_version = '22e33db27147835035f948b351a0cd78672ce15e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
