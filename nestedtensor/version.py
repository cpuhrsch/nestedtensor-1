__version__ = '0.0.1.dev202081222+510cc5b'
git_version = '510cc5b42d751fd7ac982753e5ba75fc850f9e54'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
