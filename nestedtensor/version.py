__version__ = '0.0.1.dev202072720+39804e6'
git_version = '39804e609ac487e924ed9984dd5fd16fd29fbc00'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
