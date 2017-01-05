#!/usr/bin/env python

"""
Embed PNGs and JPGs inline in HTML documents.
"""

import sys
import os
import encodings

if len(sys.argv) != 2:
    print("Usage: html_image_embedder.py document.html")
    sys.exit(0)

import bs4
import textwrap


def wrap(text):
    text = str(text, 'ascii')
    return '\n'.join(textwrap.wrap(text, 80))


html_file = sys.argv[1]
html_dir = os.path.abspath(os.path.dirname(html_file))

with open(html_file) as f:
    soup = bs4.BeautifulSoup(f.read(), "lxml")

for image_tag in soup.find_all("img"):
    src = image_tag.attrs['src']

    if not src.startswith('data'):
        _, ext = os.path.splitext(src)
        ext = ext[1:]

        image_types = {'jpg': 'jpeg',
                       'jpeg': 'jpeg',
                       'png': 'png'}

        with open(os.path.join(html_dir, src), 'rb') as image:
            image_tag.attrs['src'] = 'data:image/{};base64,{}'.format(
                image_types[ext.lower()],
                wrap(encodings.codecs.encode(image.read(), 'base64')).rstrip()
                )

print(soup)
