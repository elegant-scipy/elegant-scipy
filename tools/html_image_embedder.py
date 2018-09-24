#!/usr/bin/env python3

"""
Embed PNGs and JPGs inline in HTML documents.
"""

import sys
import os
import encodings
import urllib
import bs4
import textwrap


if len(sys.argv) != 2:
    print("Usage: html_image_embedder.py document.html")
    sys.exit(0)


def wrap(text):
    text = str(text, 'ascii')
    return '\n'.join(textwrap.wrap(text, 80))


html_file = sys.argv[1]
html_dir = os.path.abspath(os.path.dirname(html_file))

with open(html_file, 'rb') as f:
    soup = bs4.BeautifulSoup(f.read(), "lxml")

for image_tag in soup.find_all("img"):
    src = image_tag.attrs['src']

    if not src.startswith('data'):
        _, ext = os.path.splitext(src)
        ext = ext[1:]

        image_types = {'jpg': 'jpeg',
                       'jpeg': 'jpeg',
                       'png': 'png',
                       'svg': 'svg'}

        if src.startswith('http'):
            try:
                with urllib.request.urlopen(src) as response:
                    image_data = response.read()
            except urllib.error.HTTPError:
                print(f"Can't load image at {src}. Ignoring.",
                      file=sys.stderr)
        else:
            with open(os.path.join(html_dir, src), 'rb') as image:
                image_data = image.read()

        image_tag.attrs['src'] = 'data:image/{};base64,{}'.format(
            image_types[ext.lower()],
            wrap(encodings.codecs.encode(image_data, 'base64')).rstrip()
            )

sys.stdout.buffer.write(str(soup).encode('utf-8'))
