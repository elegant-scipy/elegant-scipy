# Elegant Scipy via the Executable Book Project

<font color="red"/>**WARNING: Experimental**</font>

This branch of the Elegant Scipy repo is for experimenting with the tools
being developed by the 
[Executable Book Project](https://github.com/ExecutableBookProject).

The first two chapters have been converted from the original markdown format
to [MyST](https://github.com/ExecutableBookProject/MyST-Parser), a markdown
flavor that adds `roles` and `directives` from rST with a markdown-friendly
syntax.
The [MyST-Parser](https://github.com/ExecutableBookProject/MyST-Parser) 
parses `MyST`-flavored markdown directly to the docutils AST. 
This means that sphinx can be used to build html and LaTeX/pdf output from the
original `MyST` source files.

## Installation

To install the additional dependencies required for building `MyST`-flavored
Elegant Scipy with sphinx, install the additional requirements from the
`requirements.txt` in this folder:

```
pip install -r requirements.txt
```

## Usage

Use `sphinx` to convert elegant scipy to html or pdf output.
From the `sphinx/` directory:

```bash
# For html
make html && <preferred_browser> _build/html/index.html
```

```bash
# For pdf | NOTE - LaTeX build chain must be installed on the system
make latexpdf
# Press enter to bypass LaTeX tabulary errors (see known issues)
<your_pdf_viewer> _build/latex/elegant-scipyinmyst.pdf
```

## Known Issues

The work in this branch is experimental. The `myst-parser` and other 
sphinx plugins are under active development.

 - `make html`: Sphinx warnings about multiply defined labels in bibliography
 - `make latexpdf`: LaTeX errors related to `\hline` and badly formatted table
