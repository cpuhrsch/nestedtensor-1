__version__ = '0.0.1.dev20207120+3478607'
git_version = '34786079c9e853d94004a0e17ee278b56e8a9731'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
