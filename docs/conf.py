# -- Project information -----------------------------------------------------
project = 'llumpy'
copyright = '2026, Derek Garcia'
author = 'Derek Garcia'
release = '0.0.1'
version = '0.0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # harmless even though docstrings are RST-style
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Autodoc / Autosummary ----------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_typehints = 'description'
add_module_names = False  # show `AnthropicClient`, not `llumpy.providers._anthropic.AnthropicClient`

# -- Intersphinx ---------------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# -- HTML output ---------------------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['style.css']
html_title = 'llumpy docs'
