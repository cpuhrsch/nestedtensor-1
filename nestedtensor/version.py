__version__ = '0.1.4+2458b2b'
git_version = '2458b2b536115393f49402053d22125e200dd520'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
