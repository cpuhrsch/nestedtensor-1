__version__ = '0.0.1.dev20209115+7fc0e23'
git_version = '7fc0e230915c8824a7646a75ce9aedcdcb4079a7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
