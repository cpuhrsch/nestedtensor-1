__version__ = '0.1.4+794dae1'
git_version = '794dae1fafd45be18f538a6600de9f5e5f0b4b31'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
