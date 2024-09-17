# This Makefile builds an HTML "book" from source files written in
#     Markdown.

SHELL = /bin/bash

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	SED_I = sed -i ''
else
	SED_I = sed -i''
endif

# VPATH: where to look for source files. This is a Makefile built-in
#     special variable.
VPATH = markdown

# BUILD_{HTML,NB}: where to put the output HTML files and the
#     intermediate IPython notebooks.
BUILD_HTML = ./html
BUILD_NB = ./ipynb
BUILD_HTMLBOOK = ./htmlbook
FIGURES = ./figures/generated

# TITLES: This should be an exhaustive list of all the chapters to be
#     built, and correspond to markdown filenames in the markdown
#     directory.
TITLES := preface ch1 ch2 ch3 ch4 ch5 ch6 ch7 ch8 epilogue acknowledgements

# CHS_, chs: some Makefile magic that prefixes all the titles with the
#     HTML build directory, then suffixes them with the .html
#     extension. chs then constitutes the full list of targets.
CHS_ := $(addprefix $(BUILD_HTML)/,$(TITLES))
chs: build_dirs $(addsuffix .html,$(CHS_))

ipynb/ch1.ipynb: data/counts.txt.bz2

ipynb/ch2.ipynb: data/counts.txt.bz2

ipynb/ch4.ipynb: $(FIGURES)/radar_time_signals.png $(FIGURES)/sliding_window.png

ipynb/ch7.ipynb: $(FIGURES)/optimization_comparison.png

ipynb/ch8.ipynb: data/dm6.fa

data/dm6.fa: data/dm6.fa.gz
	 gunzip -f -k $<

data/dm6.fa.gz:
	 curl --remote-name http://hgdownload.cse.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz
	 mv dm6.fa.gz data

$(FIGURES)/%.png: script/%.py $(FIGURES)
	 MPLCONFIGDIR=./.matplotlib python $< $@

# %.html: How to build an HTML file from its corresponding IPython
#     notebook.
$(BUILD_HTML)/%.html: $(BUILD_NB)/%.ipynb $(BUILD_HTML)/custom.css build_dirs
	jupyter nbconvert --to html --template classic $< --stdout > $@
	tools/html_image_embedder.py $@ > $@.embed && mv $@.embed $@

$(BUILD_HTMLBOOK)/%.html: $(BUILD_NB)/%.ipynb
	git submodule update --init --recursive
	
	mkdir -p $(BUILD_HTMLBOOK)/downloaded
	ln -sf $(PWD)/figures $(BUILD_HTMLBOOK)/figures
	ln -sf $(PWD)/images $(BUILD_HTMLBOOK)/images
	
	jupyter nbconvert --to=mdoutput --output="$(notdir $@)" --output-dir=$(BUILD_HTMLBOOK) $<
	
	TITLE=`cat $@.md | grep -e '^# ' | head -n 1 | sed 's/^# //'` ; \
	echo Chapter: $$TITLE ;\
	PYTHONIOENCODING="utf_8" tools/latex_to_mathml.py $@.md > $@.mathml && mv $@.mathml $@.md ; \
	PYTHONIOENCODING="utf_8" tools/footnote_fixer.py $@.md > $@.footnoted && cp $@.footnoted /tmp && mv $@.footnoted $@.md ; \
	${SED_I} 's/^<math/ <math/' $@.md ; \
	htmlbook -c -s $@.md -o $@ -t "$$TITLE"; \
	PYTHONIOENCODING="utf_8" tools/strip_jupyter_style.py $@ > $@.unstyled && mv $@.unstyled $@
	
	xmllint --schema OReilly_HTMLBook/schema/htmlbook.xsd --noout $@
	htmlbook -s $@.md -o $@
	rm $@.md
	
	PYTHONIOENCODING="utf_8" tools/strip_jupyter_style.py $@ > $@.unstyled && mv $@.unstyled $@ ; \
	PYTHONIOENCODING="utf_8" tools/html_image_unpacker.py $@ > $@.unpacked && mv $@.unpacked $@
	PYTHONIOENCODING="utf_8" tools/html_image_unpacker.py $@ > $@.unpacked && mv $@.unpacked $@
	PYTHONIOENCODING="utf_8" tools/wrap_callouts.py $@ > $@.tagged && mv $@.tagged $@
	
	cp $@ /tmp
	
	PYTHONIOENCODING="utf_8" tools/wrap_figure.py $@ > $@.figures && mv $@.figures $@
	PYTHONIOENCODING="utf_8" tools/caption_crunch.py $@ > $@.captions && mv $@.captions $@
	PYTHONIOENCODING="utf_8" tools/audio_objects.py $@ > $@.audio && mv $@.audio $@
	
	${SED_I} 's/\.\.\/figures/.\/figures/' $@
	${SED_I} 's/\.\.\/images/.\/images/' $@
	${SED_I} 's/data-code-language="output" data-type="programlisting"//' $@

$(BUILD_HTML)/custom.css:
	 cp style/custom.css $(BUILD_HTML)

# %.ipynb: How to build an IPython notebook from a source Markdown
#     file.
$(BUILD_NB)/%.ipynb: %.markdown style/elegant.mplstyle build_dirs
	PYTHONWARNINGS="ignore" notedown --timeout 600 --match python --run $< --output $@

# .SECONDARY: Ensure ipynb files are not deleted after being generated.
NBS_ := $(addprefix $(BUILD_NB)/,$(TITLES))
nbs: $(addsuffix .ipynb,$(NBS_))
.SECONDARY: nbs data/dm6.fa data/dm6.fa.gz

# .PHONY: Special Makefile variable specifying targets that don't
#     correspond to any actual files.
.PHONY: all build_dirs chs

# build_dirs: directories for build products
build_dirs:
	mkdir -p $(BUILD_HTML) $(BUILD_NB) $(BUILD_HTMLBOOK) $(FIGURES)
	tools/link.py data $(BUILD_NB)/data
	tools/link.py style $(BUILD_NB)/style

exercises: chs
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

htmlbook: build_dirs $(addsuffix .html, $(addprefix $(BUILD_HTMLBOOK)/,$(TITLES)))
	${SED_I} 's/data-type="chapter"/data-type="preface"/' htmlbook/preface.html
	${SED_I} 's/data-type="chapter"/data-type="afterword"/' htmlbook/epilogue.html
	${SED_I} 's/data-type="chapter"/data-type="acknowledgments"/' htmlbook/acknowledgements.html

# clean: remove intermediate products (IPython notebooks)
clean:
	 rm -rf $(BUILD_NB)

clobber: clean
	 rm -rf $(BUILD_HTML)
	 rm -rf $(FIGURES)
