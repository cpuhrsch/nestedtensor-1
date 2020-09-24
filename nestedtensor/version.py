__version__ = '0.0.1.dev20209244+9618305'
git_version = '9618305f3a6d9fa5b79b53faddb6ab3132058fbe'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
