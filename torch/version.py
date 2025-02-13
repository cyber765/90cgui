from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']
__version__ = '2.5.1'
debug = False
cuda: Optional[str] = None
git_version = 'Unknown'
hip: Optional[str] = None
