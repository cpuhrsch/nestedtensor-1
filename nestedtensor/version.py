__version__ = '0.0.1.dev20205302+f02f976'
git_version = 'f02f976e4d320c7b0071584ba41dccc8a682a696'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
