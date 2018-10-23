#!/usr/bin/env python3

import sys
import os
import re
from bs4 import BeautifulSoup


if len(sys.argv) < 2:
    print('Usage: strip_jupyter_style.py filename0.html')
    sys.exit(1)

fn = sys.argv[1]

with open(fn, encoding='utf-8') as f:
    html = f.read()

html_no_style = re.sub('<style(.*?)>(.*?)</style>',
                       '', html, flags=re.DOTALL)

print(html_no_style)
