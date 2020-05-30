__version__ = '0.0.1.dev20205304+1ba787b'
git_version = '1ba787b98f679db7fed0fc312942b48373649b18'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
