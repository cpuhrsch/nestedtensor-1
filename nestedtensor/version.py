__version__ = '0.1.4+48b72b2'
git_version = '48b72b25c80157f4a5a6ade3a166112db182cc26'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
