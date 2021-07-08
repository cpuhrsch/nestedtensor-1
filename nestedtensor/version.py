__version__ = '0.1.4+da14ac4'
git_version = 'da14ac448afc163aae14875a3702f72288f91bfd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
