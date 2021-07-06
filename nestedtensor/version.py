__version__ = '0.1.4+a72a123'
git_version = 'a72a123e503d4ca927b7eff045f0348e06eeb315'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
