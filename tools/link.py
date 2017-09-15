#!/usr/bin/env python

import sys
import os

_, src, dst = sys.argv

if not os.path.exists(dst):
    os.symlink(os.path.abspath(src), dst)
