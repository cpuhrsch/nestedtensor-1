__version__ = '0.0.1.dev202011722+07ca281'
git_version = '07ca2814355f42abeab06d4b0c2c2a872c01ab41'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
