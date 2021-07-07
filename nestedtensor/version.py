__version__ = '0.1.4+d1f78e7'
git_version = 'd1f78e7d31a406aa755d4842e94e4e159692d0a6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
