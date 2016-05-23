"""
See also:

https://github.com/Impactstory/depsy-research/blob/master/introducing_depsy.Rmd
https://github.com/Impactstory/depsy/blob/master/scripts/run_igraph.sh

Depsy tables:

('cooccurring_tags_one_way',)
('dep_nodes_ncol_cran_reverse',)
('dep_nodes_ncol_pypi_reverse',)
  -> (used_by, package)
('github_repo',)
('package_tags',)
('tags',)
('package',)
('person',)
('contribution',)

List tables with:

  SELECT table_name
  FROM information_schema.tables
  WHERE table_schema='public' AND table_type='BASE TABLE';

List columns in table with:

  select column_name from information_schema.columns where table_name='dep_nodes_ncol_pypi_reverse';

List of packages:

  select
      id,
      host,
      impact,
      impact_percentile,
      num_downloads,
      num_downloads_percentile,
      num_citations,
      num_citations_percentile,
      pagerank,
      pagerank_score,
      pagerank_percentile,
      indegree,
      neighborhood_size
  from package
  where is_academic = true";

Count nr of entries in a table:

  select count(*) from dep_nodes_ncol_pypi_reverse;

"""

import psycopg2 as pg

conn = pg.connect(
    dbname="ddstg43butl93u",
    user="u3181cudsmcf62",
    password="p16bm4ts99dvf63p9mrsbub5of7",
    host="ec2-54-83-205-154.compute-1.amazonaws.com",
    port=6482)

cur = conn.cursor()

print('Executing query on depsy db...')
cur.execute('''
  select used_by, package from dep_nodes_ncol_pypi_reverse;
''')

print('Dumping to results.txt...')
with open('results.txt', 'w') as f:
    for (used_by, package) in cur:
        f.write('{},{}\n'.format(used_by, package))
