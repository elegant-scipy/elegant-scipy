# -- Project information -----------------------------------------------------

project = 'Elegant-Scipy in Myst'
copyright = '2020, ross'
author = 'ross'

# The full version, including alpha/beta/rc tags
release = '0.0.0-dev'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",              # Supporting myst syntax
    "sphinxcontrib.bibtex",     # LaTeX-style bibliographies
    "sphinx.ext.imgconverter",  # SVG support in LaTeX
    "sphinx.ext.intersphinx",   # Intersphinx for linking to project docs
]

# Enable auto-numbering
numfig=True

# Intersphinx conf
intersphinx_mapping = {
    'scipy' : ('https://docs.scipy.org/doc/scipy/reference', None),
}

# Add any paths that contain templates here, relative to this directory.
#templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', 'README.md']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
# NOTE: For testing with jupyter-book theme sphinx ext
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for LaTeX/PDF output --------------------------------------------
