__version__ = '0.1.4+5bebd27'
git_version = '5bebd27bebd8e7b9fcb06fb0841a91c38b9a4343'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
