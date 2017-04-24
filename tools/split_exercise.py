#!/usr/bin/env python3

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

    with open(fn, encoding='utf-8') as f:
        html = f.read()
        bs = BeautifulSoup(html, 'html.parser')

    content = bs.find(id='notebook-container')
    exercises = re.findall('<!-- exercise begin -->(.*?)<!-- exercise end -->',
                           str(content), flags=re.DOTALL)

    print('{} exercises found in {}'.format(len(exercises), fn))

    if exercises:
        exercises.insert(0, '<h2>File: {}</h2>'.format(fn))
    all_exercises.extend([BeautifulSoup(e, "html.parser").prettify()
                          for e in exercises])

    html_no_exercise = re.sub('<!-- solution begin -->(.*?)<!-- solution end -->',
                              '', html, flags=re.DOTALL)

    with open(fn_solutions_stripped, 'w', encoding='utf-8') as f:
        f.write(html_no_exercise)


# We would like to replace the HTML inside `content` with the HTML of
# the exercises.  Unfortunately, Beautiful Soup does escaping upon
# assignment to `content.string`, so we just add a token here, and
# then do the replacement right before writing to file.

content.string = '{{REPLACE_ME}}'

with open(os.path.join(os.path.dirname(fn), 'exercises.html'), 'w', encoding='utf-8') as f:
    exercise_text = '\n<hr/>\n'.join(all_exercises)
    exercise_text = exercise_text.replace('<!-- solution begin', '<p><br/></p><!-- solution begin')

    # Remove CSS rule that adds newlines after inline code snippets
    bs = str(bs).replace('white-space: pre-wrap;', '')

    f.write(bs.replace('{{REPLACE_ME}}', exercise_text))
