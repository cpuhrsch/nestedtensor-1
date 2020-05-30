__version__ = '0.0.1.dev20205304+ded6245'
git_version = 'ded6245adf62c955dc5abe78b98c640ea708cd90'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
