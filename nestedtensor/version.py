__version__ = '0.1.4+eacffa6'
git_version = 'eacffa660de985d855f21132f29fbb19e5dfca02'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
