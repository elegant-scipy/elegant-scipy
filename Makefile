# This Makefile builds an HTML "book" from source files written in
#     Markdown.

# VPATH: where to look for source files. This is a Makefile built-in
#     special variable.
VPATH = markdown

# BUILD_{HTML,NB}: where to put the output HTML files and the
#     intermediate IPython notebooks.
BUILD_HTML = build_html
BUILD_NB = build_ipynb

# TITLES: This should be an exhaustive list of all the chapters to be
#     built, and correspond to markdown filenames in the markdown
#     directory.
TITLES := preface ch1 ch2 ch3 ch4 ch5 ch6 ch7 ch8 epilogue

# CHS_, chs: some Makefile magic that prefixes all the titles with the
#     HTML build directory, then suffixes them with the .html
#     extension. chs then constitutes the full list of targets.
CHS_ := $(addprefix $(BUILD_HTML)/,$(TITLES))
chs: build_dirs $(addsuffix .html,$(CHS_))

# %.html: How to build an HTML file from its corresponding IPython
#     notebook.
$(BUILD_HTML)/%.html: $(BUILD_NB)/%.ipynb
	 ipython nbconvert --to html $< --stdout > $@

# %.ipynb: How to build an IPython notebook from a source Markdown
#     file.
$(BUILD_NB)/%.ipynb: %.markdown
	 notedown --match fenced --run $< --output $@

# .SECONDARY: Ensure ipynb files are not deleted after being generated.
NBS_ := $(addprefix $(BUILD_NB)/,$(TITLES))
nbs: $(addsuffix .ipynb,$(NBS_))
.SECONDARY: nbs

# .PHONY: Special Makefile variable specifying targets that don't
#     correspond to any actual files.
.PHONY: all build_dirs

# build_dirs: directories for build products
build_dirs: $(BUILD_HTML) $(BUILD_NB)
$(BUILD_HTML):
	 mkdir -p $(BUILD_HTML)
$(BUILD_NB):
	 mkdir -p $(BUILD_NB)

# all: build the book.
all: chs

# clean: remove intermediate products (IPython notebooks)
clean:
	 rm -rf $(BUILD_NB)

clobber: clean
	 rm -rf $(BUILD_HTML)
