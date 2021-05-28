__version__ = '0.1.4+b0d6be9'
git_version = 'b0d6be932f9889aebd93ccbe6c2c4e9f858c4907'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
