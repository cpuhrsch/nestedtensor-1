__version__ = '0.1.4+6e8bd64'
git_version = '6e8bd6457ebddf3fb5095bc5814ac8bd76f33fde'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
