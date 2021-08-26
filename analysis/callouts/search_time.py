#!/usr/bin/env python3

import os
import pandas as pd
import sqlalchemy
import yaml

script_dir = os.path.dirname(os.path.realpath("__file__"))
with open('../../cfg.yml', 'r') as cfg_file:
    data = cfg_file.read()
tuner_cfg = yaml.safe_load(data)
database = tuner_cfg['database'].replace('mysql', 'mysql+pymysql')

engine = sqlalchemy.create_engine(database)
query = 'select search_time, test.name as test, tuning_run.name as tuning_run, tuning_run.start_date,' \
        ' collection_date from result' \
        ' inner join desired_result on desired_result.result_id = result.id' \
        ' inner join tuning_run on tuning_run.id = result.tuning_run_id' \
        ' inner join program_version on program_version.id = program_version_id' \
        ' inner join program on program.id = program_version.program_id' \
        ' inner join test on test.id = test_id' \
        ' where (test.name = "bayes_zcu102" or test.name = "no_samp") and program.name = "bnn"'
data = pd.read_sql_query(query, engine)

# Set the end date of each tuning run to the time the last result was collected.
data['end_date'] = data.groupby(['test', 'tuning_run'])['collection_date'].transform('max')
data['duration'] = data['end_date'] - data['start_date']
data = data.drop(columns=['end_date', 'tuning_run'])

# Determine the duration of the shortest tuning run.
min_duration = data['duration'].min()
data = data.drop(columns=['duration'])

# Give all tuning runs the same duration.
data['time'] = data['collection_date'] - data['start_date']
data = data[data['time'] <= min_duration]
data = data.drop(columns=['start_date', 'collection_date', 'time'])

# Determine the average search time for each test.
data = data.groupby(['test']).mean().reset_index()
samp_search_time = data.loc[data['test'] == '20210609_bayes_zcu102', 'search_time'].iloc[0]
opt_search_time = data.loc[data['test'] == '20210616_no_samp', 'search_time'].iloc[0]

# Show the results.
print('Search time with random sampling:', samp_search_time)
print('Search time with numerical optimization:', opt_search_time)

# Output search times to file.
with open('../callouts/search_time.tex', 'w') as output_file:
    output_file.write('\\def \\sampsearchtime {%.0f}\n' % samp_search_time)
    output_file.write('\\def \\optsearchtime {%.0f}\n' % opt_search_time)
