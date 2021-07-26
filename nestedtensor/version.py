__version__ = '0.1.4+2fbb7a7'
git_version = '2fbb7a7fd2c153f0356c48c90da4889afcc7ee86'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
