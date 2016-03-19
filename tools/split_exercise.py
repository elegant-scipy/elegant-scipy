#!/usr/bin/env python

import sys
import os
import re
from bs4 import BeautifulSoup


if len(sys.argv) != 2:
    print('Usage: split_exercise.py filename')
    sys.exit(1)


fn = sys.argv[1]
basename, ext = os.path.splitext(fn)
fn_solutions_stripped = basename + '_no_solutions' + ext
fn_exercises = basename + '_exercises' + ext

with open(fn) as f:
    html = f.read()
    bs = BeautifulSoup(html, 'html.parser')

content = bs.find(id='notebook-container')

exercises = re.findall('<!-- exercise begin -->(.*?)<!-- exercise end -->',
                       str(content), flags=re.DOTALL)

html_no_exercise = re.sub('<!-- solution begin -->(.*?)<!-- solution end -->',
                          '', html, flags=re.DOTALL)

if not exercises:
    print('No exercises found in {}'.format(fn))
    sys.exit(0)
else:
    print('Formatting {} exercises in {}'.format(len(exercises), fn))

with open(fn_solutions_stripped, 'w') as f:
    f.write(html_no_exercise)

content.string = '{{REPLACE_ME}}'

with open(fn_exercises, 'w') as f:
    f.write(str(bs).replace('{{REPLACE_ME}}', '\n<hr/>\n'.join(exercises)))
