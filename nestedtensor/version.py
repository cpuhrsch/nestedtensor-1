__version__ = '0.0.1.dev202061016+cb0f474'
git_version = 'cb0f4744e3c45eebbbb3879fc613c5eb4959ee3d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
