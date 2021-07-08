__version__ = '0.1.4+21b6bf8'
git_version = '21b6bf8f19db74da3f3c93a4cc6359771a29005a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
