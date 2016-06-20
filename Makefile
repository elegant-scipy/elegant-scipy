# This Makefile builds an HTML "book" from source files written in
#     Markdown.

SHELL = /bin/bash

# VPATH: where to look for source files. This is a Makefile built-in
#     special variable.
VPATH = markdown

# BUILD_{HTML,NB}: where to put the output HTML files and the
#     intermediate IPython notebooks.
BUILD_HTML = html
BUILD_NB = ipynb
BUILD_MD = mdout
FIGURES = figures/generated

# TITLES: This should be an exhaustive list of all the chapters to be
#     built, and correspond to markdown filenames in the markdown
#     directory.
TITLES := preface ch1 ch2 ch3 ch4 ch5 ch6 ch7 ch8 epilogue acknowledgements

# CHS_, chs: some Makefile magic that prefixes all the titles with the
#     HTML build directory, then suffixes them with the .html
#     extension. chs then constitutes the full list of targets.
CHS_ := $(addprefix $(BUILD_HTML)/,$(TITLES))
chs: build_dirs $(addsuffix .html,$(CHS_))

MD_ := $(addprefix $(BUILD_MD)/,$(TITLES))
md: build_dirs $(addsuffix .markdown,$(MD_))

ipynb/ch1.ipynb: data/counts.txt

ipynb/ch2.ipynb: data/counts.txt

ipynb/ch4.ipynb: $(FIGURES)/radar_time_signals.png $(FIGURES)/sliding_window.png

ipynb/ch8.ipynb: data/dm6.fa

.SECONDARY: data/counts.txt data/dm6.fa data/dm6.fa.gz

data/counts.txt: data/counts.txt.bz2
	 bunzip2 -d -k -f data/counts.txt.bz2

data/dm6.fa: data/dm6.fa.gz
	 gunzip -f -k $<

data/dm6.fa.gz:
	 curl --remote-name http://hgdownload.cse.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz
	 mv dm6.fa.gz data

$(FIGURES)/%.png: script/%.py $(FIGURES)
	 MPLCONFIGDIR=./.matplotlib python $< $@

# %.html: How to build an HTML file from its corresponding IPython
#     notebook.
$(BUILD_HTML)/%.html: $(BUILD_NB)/%.ipynb $(BUILD_HTML)/custom.css
	jupyter nbconvert --to html $< --stdout > $@

$(BUILD_MD)/%.markdown: $(BUILD_NB)/%.ipynb
	jupyter nbconvert --to markdown $< --stdout > $@

$(BUILD_HTML)/custom.css:
	 cp style/custom.css $(BUILD_HTML)

# %.ipynb: How to build an IPython notebook from a source Markdown
#     file.
$(BUILD_NB)/%.ipynb: %.markdown
	 notedown --timeout 600 --match python --run $< --output $@

# .SECONDARY: Ensure ipynb files are not deleted after being generated.
NBS_ := $(addprefix $(BUILD_NB)/,$(TITLES))
nbs: $(addsuffix .ipynb,$(NBS_))
.SECONDARY: nbs data/counts.txt data/dm6.fa data/dm6.fa.gz

# .PHONY: Special Makefile variable specifying targets that don't
#     correspond to any actual files.
.PHONY: all build_dirs chs

# build_dirs: directories for build products
build_dirs: $(BUILD_HTML) $(BUILD_NB)
$(BUILD_HTML):
	 mkdir -p $(BUILD_HTML)
$(BUILD_NB):
	 mkdir -p $(BUILD_NB)
$(FIGURES):
	 mkdir -p $(FIGURES)

exercises:
	./tools/split_exercise.py html/ch?.html

# all: build the book.
.DEFAULT_GOAL := all
all: chs exercises

zip: all
	DATE=`date +"%Y-%m-%d_%H-%M-%S"` ; \
	STAMP=elegant-scipy-$$DATE ; \
	ES_DIR=`pwd` ; \
	TMP_DIR=/tmp/$$STAMP ; \
	\
	rm -rf $$TMP_DIR ; \
	mkdir $$TMP_DIR ; \
	ln -s $$ES_DIR/index.html $$TMP_DIR ; \
	ln -s $$ES_DIR/html $$TMP_DIR/ ; \
	cd $$TMP_DIR/.. ; zip -r $$ES_DIR/$$STAMP.zip ./$$STAMP

# clean: remove intermediate products (IPython notebooks)
clean:
	 rm -rf $(BUILD_NB)

clobber: clean
	 rm -rf $(BUILD_HTML)
	 rm -rf $(FIGURES)
