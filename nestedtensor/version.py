__version__ = '0.0.1.dev202010262+e7bc6b1'
git_version = 'e7bc6b1234a987826a843f4e11d6bbab998653a6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
