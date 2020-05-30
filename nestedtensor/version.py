__version__ = '0.0.1.dev20205303+5f0a707'
git_version = '5f0a707c3d21a8e5ca4a9c4649d185c24b5e326b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
