__version__ = '0.0.1.dev202091218+156b195'
git_version = '156b1952d9ec48effc82f461d2e8f0c7904142f2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
