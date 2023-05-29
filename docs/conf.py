# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import re
import sys
import datetime

sys.path.append("..")

# -- Project information -----------------------------------------------------

project = "Segmentation Models"
copyright = "{}, Pavel Iakubovskii".format(datetime.datetime.now().year)
author = "Pavel Iakubovskii"


def get_version():
    sys.path.append("../segmentation_models_pytorch")
    from __version__ import __version__ as version

    sys.path.pop(-1)
    return version


version = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "autodocsumm",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# import karma_sphinx_theme
# html_theme = "karma_sphinx_theme"
import faculty_sphinx_theme

html_theme = "faculty_sphinx_theme"

# import catalyst_sphinx_theme
# html_theme = "catalyst_sphinx_theme"
# html_theme_path = [catalyst_sphinx_theme.get_html_theme_path()]

html_logo = "logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Extension configuration -------------------------------------------------

autodoc_inherit_docstrings = False
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_numpy_docstring = False

autodoc_mock_imports = [
    "torch",
    "tqdm",
    "numpy",
    "timm",
    "cv2",
    "PIL",
    "pretrainedmodels",
    "torchvision",
    "efficientnet-pytorch",
    "segmentation_models_pytorch.encoders",
    "segmentation_models_pytorch.utils",
    # 'segmentation_models_pytorch.base',
]

autoclass_content = "both"
autodoc_typehints = "description"

# --- Work around to make autoclass signatures not (*args, **kwargs) ----------


class FakeSignature:
    def __getattribute__(self, *args):
        raise ValueError


def f(app, obj, bound_method):
    if "__new__" in obj.__name__:
        obj.__signature__ = FakeSignature()


def setup(app):
    app.connect("autodoc-before-process-signature", f)


# Custom configuration --------------------------------------------------------

autodoc_member_order = "bysource"
