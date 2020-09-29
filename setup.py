from setuptools import setup

setup(
  name             = 'fastprof',
  version          = '0.2.0',
  description      = 'A fast profiling tool for likelihood-based statistical analyses',
  author           = 'N. Berger',
  author_email     = '',
  packages         = [ 'fastprof' ],
  install_requires = [ 'wheel', 'numpy', 'scipy', 'pandas', 'matplotlib', 'sphinx-rtd-theme' ],
  scripts          = [ 'utils/compute_limits.py' ]
#  entry_points     = {
#    'console_scripts' : {
#      'convert_ws.py = utils.convert_ws',
#      'compute_limits.py = utils.compute_limits',
#      'test.py = utils.test.py:main'
#      }
#    }
)
