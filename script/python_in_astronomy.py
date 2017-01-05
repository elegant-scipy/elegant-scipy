# Counting programming language mentions in astronomy papers
# ==========================================================
#
# "2016 Edition" Author: Juan Nunez-Iglesias.
# Notebook version at: https://gist.github.com/jni/3339985a016572f178d3c2f18e27ec0d
#
# Adapted from code written
# by Thomas P. Robitaille (http://mpia.de/~robitaille/) and updated
# by Chris Beaumont (https://chrisbeaumont.org/).
#
# - v0: https://nbviewer.jupyter.org/github/astrofrog/mining_acknowledgments/blob/master/Mining%20acknowledgments%20in%20ADS.ipynb
# - v1: https://nbviewer.jupyter.org/github/ChrisBeaumont/adass_proceedings/blob/master/Mining%20acknowledgments%20in%20ADS.ipynb)
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 3.0
# Unported License (http://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_US)

# A couple of years ago I came across this tweet by Chris Beaumont:
#
#   https://twitter.com/BeaumontChris/status/517412133181865984
#
# It shows Python overtaking Matlab and rapidly gaining ground on IDL in
# astronomy.
#
# I've referred to that plot a couple of times in the past, but now that I
# wanted to use it in a talk and in the book, I thought it was time to update
# it. Hence, this script.
#
# We use the `ads` Python library (https://pypi.python.org/pypi/ads),
# to simplify queries to the Astrophysics Data System
# (https://ui.adsabs.harvard.edu).


import os
import sys


if not os.path.exists(os.path.expanduser('~/.ads/dev_key')):
    print('''
To run this script, **you need to get a free API key** to allow queries to
the ADS system. Create an account here:

  https://ui.adsabs.harvard.edu/#user/account/register

then go to ADS,

  https://ui.adsabs.harvard.edu

log in, and then look for "Generate a new key" under your user profile.

Copy that key into a file called `.ads/dev_key` in your home
directory.
''')
    sys.exit(0)


if len(sys.argv) > 1:
    fn = sys.argv[1]
else:
    fn = '/tmp/python-in-astronomy.png'


# First, let's import everything we need. You can install it all using either
# conda or pip.

import os

import brewer2mpl
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from prettyplotlib.utils import remove_chartjunk
from datetime import datetime, date
import ads as ads


# Let's set some nice Matplotlib defaults. Note that there is a deprecation
# warning when setting the default color cycle, but I can't be bothered
# tracking down the fix. (It is not the simple replacement suggested by the
# deprecation message.)

mpl.rcParams['axes.color_cycle'] = brewer2mpl.get_map(
        'Paired', 'qualitative', 12).mpl_colors[1::2] + [(0.94, 0.01, 0.50)]
mpl.rcParams['figure.figsize'] = (9,6)
mpl.rcParams['font.size'] = 14


def yearly_counts(query='', years=(2000, 2017),
                  acknowledgements=False):
    """Count up how many results an individual query and year return.
    """
    if acknowledgements:
        query = 'ack:' + query
    modifiers = ' '.join(['year:%i'])
    full_query = ' '.join([query, modifiers])
    filter_query = ['database:astronomy',
                    'property:refereed']
    results = []
    for year in range(*years):
        papers = ads.SearchQuery(q=full_query % year,
                                 fq=filter_query)
        papers.execute()
        count = int(papers.response.numFound)
        total_papers = ads.SearchQuery(q=modifiers % year)
        total_papers.execute()
        total_count = int(total_papers.response.numFound)
        now = datetime.now().timetuple()
        if year == now.tm_year:
            days_in_year = date(year, 12, 31).timetuple().tm_yday
            count *= days_in_year / now.tm_yday
            total_count *= days_in_year / now.tm_yday
        results.append([year, count, total_count])
    return np.array(results)


def combine_results(res):
    """Combine related queries (such as 'MATLAB' and 'Matlab').
    """
    combined = res[0]
    for r in res[1:]:
        combined[:, 1:] += r[:, 1:]
    return combined


def trendlines(queries, norm=False):
    """Plot query results"""
    for q in queries:
        counts = queries[q]
        x = counts[:, 0]
        y = np.copy(counts[:, 1])
        if norm:
            y = y / counts[:, 2]
        plt.plot(x, y * 100, label=q, lw=4, alpha=0.8)
    plt.xlim(np.min(x), np.max(x))
    plt.xlabel('Year')
    plt.ylabel('Percent of Refereed\nPublications Mentioning')
    plt.legend(loc='upper left', frameon=False)
    remove_chartjunk(plt.gca(), ['top', 'right'])


# Dictionary that maps languages to queries. I've left some of the original
# queries commented out, but you can uncomment them if you care about those
# languages in astronomy.

# As a side note, a simple measure of how annoying a language's name is is
# given by the number of queries necessary to find its mentions.

languages = {
    'IDL': ['IDL'],
    'Python': ['Python'],
    'Matlab': ['MATLAB', 'Matlab'],
#    'Fortran': ['Fortran', 'FORTRAN'],
#    'Java': ['Java'],
#    'C': ['C programming language', 'C language',
#          'C code', 'C library', 'C module'],
#    'R': ['R programming language', 'R language',
#          'R code', 'R library', 'R module'],
}


if __name__ == "__main__":
    results = {name: combine_results([yearly_counts(query) for query in queries])
               for name, queries in languages.items()}

    trendlines(results, norm=True)
    plt.savefig(fn, dpi=600)
