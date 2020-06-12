__version__ = '0.0.1.dev20206124+d0ae413'
git_version = 'd0ae4135da0cac743d9520d13d97b282abf253c8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
