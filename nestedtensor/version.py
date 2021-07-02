__version__ = '0.1.4+e536584'
git_version = 'e536584d00f8afe8d2ce044fb9b8d1a7108a83d0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
