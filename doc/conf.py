# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sphinx_rtd_theme


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fastprof'
copyright = '2020, Nicolas Berger'
author = 'Nicolas Berger'
release = '0.3.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
  'sphinx.ext.mathjax',
  'sphinx.ext.napoleon',
  'sphinx_rtd_theme',
  'sphinxarg.ext'
]

templates_path = ['_templates']
exclude_patterns = [ '_build', 'api/utils.*' ]

language = 'en'

# The master toctree document.
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'fastprofdoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    'preamble': r'''
    \usepackage{amsmath}
    \usepackage{bm}
    ''',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'fastprof.tex', 'fastprof Documentation',
     'Nicolas Berger', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'fastprof', 'fastprof Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'fastprof', 'fastprof Documentation',
     author, 'fastprof', 'One line description of project.',
     'Miscellaneous'),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/': None}

mathjax3_config = {
    'tex': {
      'packages' : {'[+]': ['bm'] },
        'macros': {
            'vt': [r'\boldsymbol{\theta}'],
            'vm': [r'\boldsymbol{\mu}'],
            'vn': [r'\boldsymbol{n}'],
            'vD': [r'\boldsymbol{\Delta}'],
            'vQ': [r'\boldsymbol{Q}'],
            'mhat': [r'\hat{\vm}'],
            'that': [r'\hat{\vt}'],
            'dhat': [r'\boldsymbol{\delta}\that'],
            'obs' : [r'^{\text{obs}}'],
            'nom' : [r'^{\text{nom}}'],
            'ref' : [r'^{\text{ref}}'],
            'xpec': [r'^{\text{exp}}'],
            'chan': [r'_{\text{channels}}'],
            'poiss': [r'_{\text{Poisson}}'],
            'gauss': [r'_{\text{Gaussian}}'],
            'bins': [r'_{\text{bins}}'],
            'samp': [r'_{\text{samples}}'],
            'binsc': [r'_{\text{bins},c}'],
            'sampc': [r'_{\text{samples},c}'],
            'nps' : [r'_{\text{NPs}}'],
            'aux' : [r'_{\text{aux}}'],
            'pois': [r'_{\text{POIs}}'],
            'cs': [r'_{cs}'],
            'cb': [r'_{cb}'],
            'cbs': [r'_{cbs}'],
            'cbsk': [r'_{cbsk}'],
            }
        }
}

import mock,sys

MOCK_MODULES = [ 'ROOT' ]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

autodoc_mock_imports = [ 'ROOT' ]
