__version__ = '0.0.1.dev2020871+58dcf31'
git_version = '58dcf31ec3183ffd0d2e97dc70f86a1f5b9aba22'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
