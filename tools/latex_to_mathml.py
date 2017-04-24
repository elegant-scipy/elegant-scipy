#!/usr/bin/env python3

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
    print("Usage: latex_to_mathml.py document.md")
    sys.exit(0)


html_file = sys.argv[1]
html_dir = os.path.abspath(os.path.dirname(html_file))

with open(html_file, 'rb') as f:
    data = f.read().decode('utf-8')


def pipe(command, stdin_data):
    ps = subprocess.Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout_data, stderr_data = ps.communicate(input=stdin_data.encode('utf-8'))

    if ps.returncode != 0:
        sys.stderr.write(stderr_data.decode('utf-8'))
        sys.stderr.write('Saving stdin to /tmp/pipe.log\n')
        with open('/tmp/pipe.log', 'w') as f:
            f.write(stdin_data)
        sys.exit(-1)

    return stdout_data.decode('utf-8')


def latex_to_mathml(latex_str):
    mathml = pipe(['blahtexml', '--mathml'], latex_str)
    mathml = ''.join(mathml.split('\n')[1:-2])
    mathml = mathml.replace('<mathml>', '<math xmlns="http://www.w3.org/1998/Math/MathML">')
    mathml = mathml.replace('</mathml>', '</math>')
    mathml = mathml.replace('<markup>', '')
    mathml = mathml.replace('</markup>', '')
    return mathml


def math_wrap(matchobj):
    # Ignore code blocks
    if matchobj.group(1) is None:
        return matchobj.group(0)

    math = matchobj.group(1)
    return latex_to_mathml(math)

def equation_wrap(matchobj):
    # Ignore code blocks
    if matchobj.group(1) is None:
        return matchobj.group(0)

    math = matchobj.group(1)

    label = []
    def gather_labels(matchobj):
        label.append(matchobj.group(1))

    math = re.sub('\\\label{(.*?)}', gather_labels, math, flags=re.DOTALL)

    if label:
        label = '<h5>{}</h5>'.format(label[0])
    else:
        label = ''

    return '<div data-type="equation">{}{}</div>'.format(label, latex_to_mathml(math))


mathml_data = re.sub('```python[^`]+```|\$\$(.*?)\$\$', equation_wrap, data, flags=re.DOTALL)
mathml_data = re.sub('```python[^`]+```|\$(.*?)\$', math_wrap, mathml_data, flags=re.DOTALL)
print(mathml_data)
