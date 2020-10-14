from setuptools import setup, Command
import os

name    = 'fastprof'
version = '0.2.1'

setup(
  name             = name,
  version          = version,
  description      = 'A fast profiling tool for likelihood-based statistical analyses',
  author           = 'N. Berger',
  author_email     = 'nicolas.berger@cern.ch',
  packages         = [ 'fastprof', 'root_import' ],
  install_requires = [ 'numpy', 'scipy', 'pandas', 'matplotlib', 'sphinx-rtd-theme', 'sphinx-argparse' ],
  scripts          = [
    'root_import/convert_ws.py',
    'root_import/fit_ws.py',
    'utils/plot_valid.py',
    'utils/check_model.py',
    'utils/fit_fast.py',
    'utils/plot.py',
    'utils/make_inputs.py',
    'utils/compute_limits.py',
    'utils/dump_samples.py',
    'utils/iterate.py',
    'utils/collect_results.py'
    ],
  setup_requires = [ 'wheel', 'sphinx' ],
)

