__version__ = '0.1.4+f0f3d3d'
git_version = 'f0f3d3d36adb445f0f79765ce4a064455247a8e5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
