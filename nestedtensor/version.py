__version__ = '0.0.1.dev202033023+5f9bff8'
git_version = '5f9bff818cfb2e826e987f36266d5ffc9dfb2fa0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
