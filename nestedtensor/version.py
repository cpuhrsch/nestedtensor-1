__version__ = '0.0.1.dev20204922+11db089'
git_version = '11db08937d6b19dc0968cfb3ad24c02e7d27ab34'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
