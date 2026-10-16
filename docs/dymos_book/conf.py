"""Sphinx configuration for the Dymos documentation."""

project = 'Dymos'
author = 'The Dymos Development Team'
copyright = '2026, The Dymos Development Team'
root_doc = 'index'

extensions = [
    'myst_nb',
    # This line is a workaround for https://github.com/jupyter/nbconvert/issues/528
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'numpydoc',
    'sphinxcontrib.bibtex',
    'sphinx_copybutton',
    'sphinx_design',
]

# MyST / myst-nb settings
myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'dollarmath',
    'linkify',
    'substitution',
]

# Do not re-execute notebooks; use the pre-executed outputs from papermill
nb_execution_mode = 'off'
nb_output_stderr = 'show'

# Theme
html_theme = 'sphinx_book_theme'
html_title = 'Dymos'
html_logo = 'dymos_logo.png'
html_static_path = ['_static']
html_css_files = ['myfile.css']
html_theme_options = {
    'repository_url': 'https://github.com/OpenMDAO/dymos',
    'repository_branch': 'master',
    'path_to_docs': 'docs/dymos_book',
    'use_issues_button': True,
    'use_repository_button': True,
    'launch_buttons': {
        'colab_url': 'https://colab.research.google.com',
    },
    'navigation_with_keys': False,
}

# Number figures, tables, and code blocks (matches the jupyter-book default)
numfig = True

# BibTeX
bibtex_bibfiles = ['references.bib']

# Autodoc / autosummary
autosummary_generate = True
numpydoc_class_members_toctree = False

# Exclude patterns
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**/.ipynb_checkpoints',
    'template.ipynb',
    'test',
    'requirements.txt',
]
