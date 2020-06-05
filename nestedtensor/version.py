__version__ = '0.0.1.dev2020657+05fc9f8'
git_version = '05fc9f8f8b8fccc73dd75ddf4bb2f17faf0492aa'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
