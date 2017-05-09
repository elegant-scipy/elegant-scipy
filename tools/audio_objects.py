#!/usr/bin/env python3

"""
Display an image instead of audio objects
"""

import sys
import os
import re
import bs4

import subprocess
from subprocess import PIPE


if len(sys.argv) != 2:
    print("Usage: audio_objects.py document.html")
    sys.exit(0)

html_file = sys.argv[1]
html_dir = os.path.abspath(os.path.dirname(html_file))

with open(html_file, 'rb') as f:
    data = f.read().decode('utf-8')


soup = bs4.BeautifulSoup(data, "lxml")

def wrap(to_wrap, wrap_in):
    contents = to_wrap.replace_with(wrap_in)
    wrap_in.append(contents)

regexp = re.compile('.*audio controls="controls".*')

audio_outputs = soup.find_all(text=regexp)
for output in audio_outputs:
    img_tag = soup.new_tag('img')
    img_tag['src'] = './images/audio_control.png'
    output.parent.replace_with(img_tag)

print(soup)
