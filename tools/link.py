#!/usr/bin/env python

import sys
import os

_, src, dst = sys.argv

if not os.path.exists(dst):
    os.symlink(os.path.relpath(src, start=os.path.dirname(dst)),
               dst)
