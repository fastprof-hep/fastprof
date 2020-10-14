from setuptools import setup, Command
import os
#import sphinx.ext.apidoc
#from sphinx.application import Sphinx

name    = 'fastprof'
version = '0.2.1'

#class SphinxCommand(Command):
    #user_options = []
    #description = 'sphinx'

    #def initialize_options(self):
        #pass

    #def finalize_options(self):
        #pass

    #def run(self):
        ## metadata contains information supplied in setup()
        #metadata = self.distribution.metadata
        ## package_dir may be None, in that case use the current directory.
        #src_dir = (self.distribution.package_dir or {'': ''})['']
        #src_dir = os.path.join(os.getcwd(),  src_dir)
        ## Run sphinx by calling the main method, '--full' also adds a conf.py
        #print('Running apidoc')
        #sphinx.ext.apidoc.main(
            #[ '-fMeT',  '-o',  os.path.join('doc', 'api'), '.', 'setup.py' ])
##            ['', '--full', '-H', metadata.name, '-A', metadata.author,
##             '-V', metadata.version, '-R', metadata.version,
##             '-o', os.path.join('doc'), src_dir])
        ## build the doc sources
        #print('Running sphinx')
        #srcdir = 'doc'
        #build_dir = os.path.join('build', 'sphinx')
        #Sphinx(srcdir = srcdir, confdir = srcdir, outdir = build_dir, doctreedir = os.path.join(build_dir, 'doctrees'), buildername = 'html').build()

setup(
  name             = name,
  version          = version,
  description      = 'A fast profiling tool for likelihood-based statistical analyses',
  author           = 'N. Berger',
  author_email     = 'nicolas.berger@cern.ch',
  packages         = [ 'fastprof', 'root_import' ],
  install_requires = [ 'numpy', 'scipy', 'pandas', 'matplotlib', 'sphinx-rtd-theme', 'sphinx-argparse' ],
  #cmdclass         = { 'sphinx' :  SphinxCommand },
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
  setup_requires = [ 'wheel', 'sphinx'],
  entry_points = {
    'distutils.commands': [
      'sphinx = example_module:Sphinx'
    ]
  }
)

