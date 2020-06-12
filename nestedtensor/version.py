__version__ = '0.0.1.dev202061123+bbaf74e'
git_version = 'bbaf74e298fcd3b352427d40b4dafb0d6290a5ae'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
