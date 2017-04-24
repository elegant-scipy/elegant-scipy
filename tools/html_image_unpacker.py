#!/usr/bin/env python3

"""
Unpack remote PNGs and JPGs from HTML documents.
"""

import sys
import os
import urllib
import bs4


if len(sys.argv) != 2:
    print("Usage: html_image_unpacker.py document.html")
    sys.exit(0)


html_file = sys.argv[1]
html_dir = os.path.abspath(os.path.dirname(html_file))

with open(html_file, 'rb') as f:
    soup = bs4.BeautifulSoup(f.read(), "lxml")

for image_tag in soup.find_all("img"):
    src = image_tag.attrs['src']

    if src.startswith('http'):
        root, fn = os.path.split(src)

        new_url = 'downloaded/{}'.format(fn)

        with urllib.request.urlopen(src) as response:
            print('Downloading external media: {}'.format(src))

            with open(os.path.join(html_dir, new_url), 'wb') as image_file:
                image_file.write(response.read())

        image_tag.attrs['src'] = new_url

sys.stdout.buffer.write(str(soup).encode('utf-8'))
