from setuptools import setup

setup(
  name             = 'fastprof',
  version          = '0.2.1',
  description      = 'A fast profiling tool for likelihood-based statistical analyses',
  author           = 'N. Berger',
  author_email     = 'nicolas.berger@cern.ch',
  packages         = [ 'fastprof', 'root_import', 'utils' ],
  install_requires = [ 'numpy', 'scipy', 'pandas', 'matplotlib', 'sphinx-rtd-theme', 'sphinx-argparse', 'mock' ],
  entry_points = {
    'console_scripts': [
      'convert_ws.py      = root_import.convert_ws:run',
      'fit_ws.py          = root_import.fit_ws:run',
      'plot.py            = utils.plot:run',
      'plot_valid.py      = utils.plot_valid:run',
      'check_model.py     = utils.check_model:run',
      'fit_fast.py        = utils.fit_fast:run',
      'make_inputs.py     = utils.make_inputs:run',
      'compute_limits.py  = utils.compute_limits:run',
      'dump_samples.py    = utils.dump_samples:run',
      'iterate.py         = utils.iterate:run',
      'collect_results.py = utils.collect_results:run'
    ],
  },
  scripts          = [
    ],
  setup_requires = [ 'wheel', 'sphinx' ],
)

