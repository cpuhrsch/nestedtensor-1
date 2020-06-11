__version__ = '0.0.1.dev20206112+277e409'
git_version = '277e4091009b59dcf3d18330c15fcfde8df6321c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
