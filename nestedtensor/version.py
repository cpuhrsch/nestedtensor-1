__version__ = '0.0.1.dev202061021+f04157a'
git_version = 'f04157a32f642f3f5a6f6ff86df6ce6fec095830'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
