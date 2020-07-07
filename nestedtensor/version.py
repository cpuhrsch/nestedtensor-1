__version__ = '0.0.1.dev20207716+cde5125'
git_version = 'cde5125aef380ab5ecc50b161579a65c9d3f89c9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
