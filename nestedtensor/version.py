__version__ = '0.0.1.dev20202120+8bb0294'
git_version = '8bb02943a990a7cb49b4ab39ba553b6b2cf71cda'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
