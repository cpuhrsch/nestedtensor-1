__version__ = '0.0.1.dev202061720+9245f25'
git_version = '9245f2534f81b9b946d580e2832cbae7f89a33a2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
