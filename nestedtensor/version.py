__version__ = '0.0.1.dev20207117+80b4735'
git_version = '80b47354eb8fde9a5b8047479c7e6e6768d6cebd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
