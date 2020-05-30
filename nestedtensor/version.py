__version__ = '0.0.1.dev20205305+c8d57d4'
git_version = 'c8d57d4f0fb99c0c4c78e2861310670e3fdbdff6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
