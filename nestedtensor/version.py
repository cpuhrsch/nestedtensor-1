__version__ = '0.1.4+a9952c5'
git_version = 'a9952c5a631b51407eda6b6bf291652615340a9a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
