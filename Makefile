VPATH = markdown
BUILD_HTML = build_html
BUILD_NB = build_ipynb

TITLES := preface ch2
CHS_ := $(addprefix $(BUILD_HTML)/,$(TITLES))

chs: $(addsuffix .html,$(CHS_))
$(BUILD_HTML)/%.html: $(BUILD_NB)/%.ipynb
	 ipython nbconvert --to html $< --stdout > $@

$(BUILD_NB)/%.ipynb: %.markdown
	 notedown --match fenced --run $< > $@

.PHONY: all build_dirs

build_dirs:
	 mkdir -p build_ipynb
	 mkdir -p build_html

all: build_dirs chs
