__version__ = '0.0.1.dev20204922+8cf8c70'
git_version = '8cf8c70f7a8e61ae084d6ce91968c377e212ede0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
