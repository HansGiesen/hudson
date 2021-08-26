#!/usr/bin/env python3

import os
import pandas as pd
import sqlalchemy
import yaml
from gzip import zlib
from pickle import loads

script_dir = os.path.dirname(os.path.realpath("__file__"))
with open('../../cfg.yml', 'r') as cfg_file:
    data = cfg_file.read()
tuner_cfg = yaml.safe_load(data)
database = tuner_cfg['database'].replace('mysql', 'mysql+pymysql')

engine = sqlalchemy.create_engine(database)
query = 'select result.state, tuning_run.name, tuning_run.start_date, collection_date, args from result' \
        ' inner join configuration on configuration.id = result.configuration_id' \
        ' inner join tuning_run on tuning_run.id = result.tuning_run_id' \
        ' inner join program_version on program_version.id = program_version_id' \
        ' inner join program on program.id = program_version.program_id' \
        ' inner join test on test.id = test_id' \
        ' where test.name = "pipe_zcu102" and tuning_run.name like "bnn_4x1.1x2.1x2_%%" and fidelity = 3'
data = pd.read_sql_query(query, engine)

# Get command line arguments of each tuning run.
args = data['args'].transform(zlib.decompress)
data['args'] = args.transform(lambda field: loads(field, encoding='latin1'))

# Get the number of seed configurations.
seed_cnt = data['args'].iloc[0].bayes_seed_cnt
data = data.drop(columns=['args'])

# I accidentally forgot to set the seed for some OpenTuner runs, but I can extract it from the name.
data['seed'] = data['name'].str.extract(r'.*_(\d+)$').astype(int)
data = data.drop(columns=['name'])

# Ignore failed builds.
data = data[data['state'] == 'OK']
data = data.drop(columns=['state'])

# Compute the tuning time in days.
data['time'] = (data['collection_date'] - data['start_date']).dt.total_seconds()
data = data.drop(columns=['start_date', 'collection_date'])

# Order the data by time.
data = data.sort_values('time')

# Get the last time of the last configuration that was part of the initialization.
data = data.groupby('seed').nth(seed_cnt - 1)

# Compute the average initialization time in hours.
avg_init_time = data['time'].mean() / 3600.0

# Output the average initialization time.
print('Average initialization time: %.0f hours' % avg_init_time)

# Output callouts to file.
with open('../callouts/init_time.tex', 'w') as output_file:
    output_file.write('\\def \\avginittime {%.0f}\n' % avg_init_time)
