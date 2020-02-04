__version__ = '0.0.1.dev2020246+8c56c90'
git_version = '8c56c90488d49919fbe4e061a54a99c9131b7049'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
