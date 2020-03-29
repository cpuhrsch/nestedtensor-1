__version__ = '0.0.1.dev202032920+7e56fa5'
git_version = '7e56fa53b3a4f93017bc840acc7a27f37a41fe01'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
