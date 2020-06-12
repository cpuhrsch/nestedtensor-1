__version__ = '0.0.1.dev20206125+651ff58'
git_version = '651ff587437eea63badef36b3618226015f6af3d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
