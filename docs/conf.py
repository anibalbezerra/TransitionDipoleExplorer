import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'TransitionDipoleExplorer'
copyright = '2024, Anibal T Bezerra'
author = 'Anibal T Bezerra'

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']

exclude_patterns = []

html_theme = 'alabaster'

html_static_path = ['_static']
