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


def translate_tag(matchobj):
    tag = matchobj.group(1)
    return '<{}>'.format(tag)


with_tags = re.sub('<!-- +tag: +(.*?) +-->', translate_tag, data)

# Consider the following scenario:
#
# <!-- tag: aside -->
# # My section
# Some text
# <!-- tag: /aside -->
# # Another section
#
# htmlbook.js will translate this to:
#
# <!-- tag: aside -->
# <section>
# <h1>My section</h1>
# Some text
# <!-- tag: /aside -->
# </section>
#
# i.e., the closing aside tag is placed inside the section.
#
# To rectify the situation, we use BeautifulSoup to fix the
# order of tags as best it can.

soup = bs4.BeautifulSoup(with_tags, "html5lib")
print(soup)
