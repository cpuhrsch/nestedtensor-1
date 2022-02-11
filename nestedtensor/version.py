__version__ = '0.1.4+035109c'
git_version = '035109c3569bd276121c1a80b8b8ea65327e821b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
