#!/usr/bin/env python

"""
Translate dollar expressions inside HTML files to MathML
"""

import sys
import os
import re
import html

import subprocess
from subprocess import PIPE


if len(sys.argv) != 2:
    print("Usage: latex_to_mathml.py document.html")
    sys.exit(0)


html_file = sys.argv[1]
html_dir = os.path.abspath(os.path.dirname(html_file))

with open(html_file, 'rb') as f:
    data = f.read().decode('utf-8')

# Temporary workaround for figure captions containing math.  Destroy them.
data = re.sub('alt=\'.*?\'', '', data, flags=re.DOTALL)

def math_wrap(matchobj):
    math = matchobj.group(1)
#    math = html.escape(math)

    return '''
<div xmlns:b="http://gva.noekeon.org/blahtexml" b:inline="{}"/>
'''.format(math)

html_with_hooks = re.sub('\$(.*?)\$', math_wrap, data, flags=re.DOTALL)

ps = subprocess.Popen(('blahtexml', '--xmlin'), stdin=PIPE, stdout=PIPE, stderr=PIPE)
stdout_data, stderr_data = ps.communicate(input=html_with_hooks.encode('utf-8'))

if ps.returncode != 0:
    sys.stderr.write(stderr_data.decode('utf-8'))
    sys.stderr.write(stdout_data.decode('utf-8'))
    sys.exit(-1)


post_mathml = stdout_data.decode('utf-8')

def inner_only(matchobj):
    return matchobj.group(1)

post_mathml = re.sub(
    '<div xmlns:b="http://gva.noekeon.org/blahtexml">(.*?)</div>',
    inner_only, post_mathml, flags=re.DOTALL)

print(post_mathml.replace('<?xml version="1.0" encoding="UTF-8"?>', ''))
