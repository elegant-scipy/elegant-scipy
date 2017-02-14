#!/usr/bin/env python

"""
Insert HTMLBook tags where needed.
"""

import sys
import os
import re
import bs4

import subprocess
from subprocess import PIPE


if len(sys.argv) != 2:
    print("Usage: wrap_callouts.py document.html")
    sys.exit(0)

html_file = sys.argv[1]
html_dir = os.path.abspath(os.path.dirname(html_file))

with open(html_file, 'rb') as f:
    data = f.read().decode('utf-8')


soup = bs4.BeautifulSoup(data, "lxml")

def wrap(to_wrap, wrap_in):
    contents = to_wrap.replace_with(wrap_in)
    wrap_in.append(contents)


figures = soup.body.find_all('img', attrs={'alt': re.compile('^(?!png)')})

for fig in figures:
    figure_tag = soup.new_tag('figure')

    figure_caption = soup.new_tag('figcaption')
    figure_caption.string = fig.attrs['alt']
    figure_tag.append(figure_caption)

    wrap(fig, figure_tag)

print(soup)
