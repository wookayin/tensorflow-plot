# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = tfplot
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Local sphinx build
# Auto-refresh whenever file is updated; requires fswatch (brew) and http-server (npm)
dev:
	trap 'pkill -9 -g $$$$' SIGINT SIGTERM EXIT;
	http-server _build/html/ &
	fswatch --event Updated ../tfplot/ ./ --exclude _build | xargs -n 1 -I{} make html;
	wait;

.PHONY: help dev Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	PYTHONPATH=".." $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
