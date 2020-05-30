__version__ = '0.0.1.dev20205304+1bcc1fb'
git_version = '1bcc1fbcb6047f82ae3ae591066e330b3666971a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
