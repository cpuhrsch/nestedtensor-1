__version__ = '0.1.4+3613746'
git_version = '36137469e685f25837a2441397862e92c18457eb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
