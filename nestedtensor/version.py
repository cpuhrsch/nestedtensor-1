__version__ = '0.1.4+bb81821'
git_version = 'bb818218ecf32c7f799f54b9a55a807b1368e7be'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
