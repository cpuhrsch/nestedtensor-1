__version__ = '0.0.1.dev202091117+157eb4a'
git_version = '157eb4a7538dd9cd01a1c94e785f137acb9355de'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
