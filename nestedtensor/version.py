__version__ = '0.0.1.dev20206124+0b7a937'
git_version = '0b7a9377ba1d9def8695c265ab1766f3ea66dbd3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
