#!/usr/bin/env python

"""
Footnotes to HTMLBook
"""

import sys
import os
import re

import subprocess
from subprocess import PIPE


if len(sys.argv) != 2:
    print("Usage: footnote_fixer.py document.markdown")
    sys.exit(0)

md_file = sys.argv[1]

with open(md_file, 'rb') as f:
    data = f.read().decode('utf-8')


footnotes = {}

def footnote_index(match):
    fid, ftext = match.group(1), match.group(2)

    ftext = re.sub(' +', ' ', ftext)
    ftext = re.sub('^ ', '', ftext, flags=re.MULTILINE)

    footnotes[fid] = ftext

    return ''

data = re.sub('^\[\^(.*?)\]: (.*?)\n\n', footnote_index,
              data, flags=re.DOTALL|re.MULTILINE)


def insert_footnote(fid_match):
    fid = fid_match.group(1)

    return '<span data-type="footnote">\n{}</span>'.format(footnotes[fid])

print(re.sub('\[\^(.*?)\][^:]', insert_footnote, data))
