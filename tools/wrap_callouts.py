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
    print("Usage: tag_translator.py document.html")
    sys.exit(0)

html_file = sys.argv[1]
html_dir = os.path.abspath(os.path.dirname(html_file))

with open(html_file, 'rb') as f:
    data = f.read().decode('utf-8')


soup = bs4.BeautifulSoup(data, "lxml")

def wrap(to_wrap, wrap_in):
    contents = to_wrap.replace_with(wrap_in)
    wrap_in.append(contents)


callouts = soup.body.find_all(text=re.compile('{.callout}'))
blockquotes = [tag.find_parent('blockquote') for tag in callouts]

for tag in blockquotes:
    aside = soup.new_tag('aside')
    aside['data-type'] = 'sidebar'
    wrap(tag, aside)

soup = str(soup).replace('{.callout}', '')

print(soup)
