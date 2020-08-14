__version__ = '0.0.1.dev202081419+db99164'
git_version = 'db99164667dc88c1507d01d259a7df98439af504'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
