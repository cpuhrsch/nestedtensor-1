__version__ = '0.1.4+d097ccc'
git_version = 'd097ccc95f468c2e5fce7ba6d7a0f2ffc6bafd7b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
