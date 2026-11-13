# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = pixi run -e documentation python -m sphinx
SPHINXPROJ    = bilby
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile html-api

# Build the narrative docs plus only a subset of the API reference, e.g.,
#   make html-api API=bilby.core.prior,bilby.gw.conversion
# Use API=none to skip the API reference entirely. A separate build directory
# is used so the cached environment for the full build is not invalidated.
API ?= none
html-api:
	@BILBY_DOCS_API="$(API)" $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)/api-subset" $(SPHINXOPTS) $(O)
	@echo "Open $(BUILDDIR)/api-subset/html/index.html"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)