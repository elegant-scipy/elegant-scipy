#!/usr/bin/env python

import sys
import os
import re
from bs4 import BeautifulSoup


if len(sys.argv) < 2:
    print('Usage: split_exercise.py filename0.html [filename1.html] ...')
    sys.exit(1)


all_exercises = []

for fn in sys.argv[1:]:
    basename, ext = os.path.splitext(fn)
    fn_solutions_stripped = basename + '_no_solutions' + ext

    with open(fn) as f:
        html = f.read()
        bs = BeautifulSoup(html, 'html.parser')

    content = bs.find(id='notebook-container')
    exercises = re.findall('<!-- exercise begin -->(.*?)<!-- exercise end -->',
                           str(content), flags=re.DOTALL)
    all_exercises.extend(exercises)
    print('{} exercises found in {}'.format(len(exercises), fn))

    html_no_exercise = re.sub('<!-- solution begin -->(.*?)<!-- solution end -->',
                              '', html, flags=re.DOTALL)

    with open(fn_solutions_stripped, 'w') as f:
        f.write(html_no_exercise)


# We would like to replace the HTML inside `content` with the HTML of
# the exercizes.  Unfortunately, Beautiful Soup does escaping upon
# assignment to `content.string`, so we just add a token here, and
# then do the replacement right before writing to file.
content.string = '{{REPLACE_ME}}'

with open(os.path.join(os.path.dirname(fn), 'exercises.html'), 'w') as f:
    f.write(str(bs).replace('{{REPLACE_ME}}', '\n<hr/>\n'.join(exercises)))
