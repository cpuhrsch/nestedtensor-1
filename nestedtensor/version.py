__version__ = '0.0.1.dev2020423+a2414aa'
git_version = 'a2414aa19c99baf8b46346392261ccd51c616fb9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
