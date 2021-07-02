__version__ = '0.1.4+7d9d412'
git_version = '7d9d412f36b499e03c3ec30eb11cf7715a4ec256'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
