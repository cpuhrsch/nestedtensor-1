__version__ = '0.0.1.dev2020720+1041b74'
git_version = '1041b74be0f09ce7f3e9100b3694e7d5de9f4e9a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
