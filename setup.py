from setuptools import setup

setup(
  name             = 'fastprof',
  version          = '0.2.0',
  description      = 'A fast profiling tool for likelihood-based statistical analyses',
  author           = 'N. Berger',
  author_email     = '',
  packages         = [ 'fastprof' ],
  install_requires = [ 'wheel', 'numpy', 'scipy', 'pandas', 'matplotlib', 'sphinx-rtd-theme' ],
  scripts          = [ 
    'utils/convert_ws.py',
    'utils/fit_ws.py',
    'utils/plot_valid.py',
    'utils/check_model.py',
    'utils/fit_fast.py',
    'utils/plot.py',
    'utils/make_inputs.py',
    'utils/compute_limits.py',
    'utils/dump_samples.py',
    'utils/collect_results.py'
    ]
#  entry_points     = {
#    'console_scripts' : {
#      'convert_ws.py = utils.convert_ws',
#      'compute_limits.py = utils.compute_limits',
#      'test.py = utils.test.py:main'
#      }
#    }
)
