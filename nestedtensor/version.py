__version__ = '0.0.1.dev20206112+be9ba82'
git_version = 'be9ba825f9968309913e27b4cbd974661d3ce4ba'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
