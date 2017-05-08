#!/usr/bin/env python3

"""
Insert captions around output figures
"""

import sys
import os
import re
import bs4

import subprocess
from subprocess import PIPE


if len(sys.argv) != 2:
    print("Usage: caption_crunch.py document.html")
    sys.exit(0)

html_file = sys.argv[1]
html_dir = os.path.abspath(os.path.dirname(html_file))

with open(html_file, 'rb') as f:
    data = f.read().decode('utf-8')


soup = bs4.BeautifulSoup(data, "lxml")

def wrap(to_wrap, wrap_in):
    contents = to_wrap.replace_with(wrap_in)
    wrap_in.append(contents)

def is_caption(node):
    if isinstance(node, bs4.Comment):
        return node.strip().startswith('caption')

regexp = re.compile('.*text="(.*?)"', flags=re.DOTALL)

comments = soup.find_all(string=is_caption)
for comment in comments:
    match = regexp.search(comment.strip())
    if match:
        caption = match.group(1)

        # Find the output image div and wrap it inside a figure
        # tag
        for tag in comment.previous_elements:
            if tag.name == 'img':
                figure_tag = soup.new_tag('figure')
                figure_caption = soup.new_tag('figcaption')
                figure_caption.append(bs4.BeautifulSoup(caption, 'html.parser'))
                figure_tag.append(figure_caption)

                wrap(tag, figure_tag)

                break
        else:
            raise RuntimeError('Could not find output block for caption: {}'.format(caption))

print(soup)
