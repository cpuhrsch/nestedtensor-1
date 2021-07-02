__version__ = '0.1.4+39dbb43'
git_version = '39dbb432a9f1b1561aa6fb73dcfc50a4843f94d2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
