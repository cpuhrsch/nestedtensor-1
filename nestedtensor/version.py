__version__ = '0.0.1.dev20205303+4486a34'
git_version = '4486a34af9eb97640d667bb58f049ce95e1e62a0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
