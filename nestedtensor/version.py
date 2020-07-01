__version__ = '0.0.1.dev20207120+484a7d6'
git_version = '484a7d677b0c2b2a015d0fdd0098efadd35260b5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
