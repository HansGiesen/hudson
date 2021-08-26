#!/usr/bin/env python3

import os
import pandas as pd
import re
import sqlalchemy
import yaml
from gzip import zlib
from pickle import loads

tuning_time = 3.0

def extract_parallelism(data):
    if hasattr(data, 'core_map'):
        procs = re.match(r'(\d+)x\d+\.(\d+)x\d+\.(\d+)x\d+', data.core_map).groups()
        return sum([int(cnt) for cnt in procs])
    else:
        return data.parallelism

script_dir = os.path.dirname(os.path.realpath("__file__"))
with open('../../cfg.yml', 'r') as cfg_file:
    data = cfg_file.read()
tuner_cfg = yaml.safe_load(data)
database = tuner_cfg['database'].replace('mysql', 'mysql+pymysql')
    
engine = sqlalchemy.create_engine(database)
query = 'select build_time, tuning_run.name, tuning_run.start_date, collection_date, args from result' \
        ' inner join desired_result on desired_result.result_id = result.id' \
        ' inner join tuning_run on tuning_run.id = result.tuning_run_id' \
        ' inner join program_version on program_version.id = program_version_id' \
        ' inner join program on program.id = program_version.program_id' \
        ' inner join test on test.id = test_id' \
        ' where test.name = "opentuner_zcu102" and program.name = "bnn" or' \
        '       test.name = "pipe_zcu102" and tuning_run.name like "bnn_4x1.1x2.1x2_%%"'
data = pd.read_sql_query(query, engine)

# Get command line arguments of each tuning run.
args = data['args'].transform(zlib.decompress)
args = args.transform(lambda field: loads(field, encoding='latin1'))
data = data.drop(columns=['args'])

# Get the search technique of each tuning run.
data['technique'] = args.transform(lambda field: field.technique[0])

# Get the number of threads of each tuning run.
data['parallelism'] = args.apply(extract_parallelism)

# Extract the seed from the tuning run name.
data['seed'] = data['name'].str.extract(r'.*_(\d+)$').astype(int)
data = data.drop(columns=['name'])

# Compute the tuning time in days.
data['time'] = (data['collection_date'] - data['start_date']).dt.total_seconds() / 86400.0
data = data.drop(columns=['start_date', 'collection_date'])

# Make sure all runs last 3 days.
data = data[data['time'] < tuning_time]
data = data.drop(columns=['time'])

# Compute the total build time in days.
data = data.groupby(['technique', 'seed', 'parallelism']).sum() / 86400.0
data = data.reset_index()

# Compute the core utilization.
data['core_util'] = data['build_time'] / (tuning_time * data['parallelism'])
core_util_opentuner = data.loc[data['technique'] == 'AUCBanditMetaTechniqueA', 'core_util'].mean()
core_util_hudson = data.loc[data['technique'] == 'PipelinedMultiFidBayes', 'core_util'].mean()

# Show the core utilization.
print('Utilization of build time with OpenTuner:', core_util_opentuner)
print('Utilization of build time with HuDSoN:', core_util_hudson)

# Output callouts to file.
with open('../callouts/idle_time.tex', 'w') as output_file:
    output_file.write('\\def \\coreutilopentuner {%0.0f}\n' % (100.0 * core_util_opentuner))
    output_file.write('\\def \\coreutilhudson {%0.0f}\n' % (100.0 * core_util_hudson))
