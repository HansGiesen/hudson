#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sqlalchemy
import yaml
from gzip import zlib
from pickle import loads

param_cnt = 10
max_time = 3

script_dir = os.path.dirname(os.path.realpath("__file__"))
with open('../../cfg.yml', 'r') as cfg_file:
    data = cfg_file.read()
tuner_cfg = yaml.safe_load(data)
database = tuner_cfg['database'].replace('mysql', 'mysql+pymysql')

engine = sqlalchemy.create_engine(database)

query = 'select configuration.id, data, result.state, run_time, collection_date, start_date, proc_freq from result' \
        ' inner join configuration on result.configuration_id = configuration.id' \
        ' inner join tuning_run on tuning_run.id = result.tuning_run_id' \
        ' inner join program_version on program_version.id = program_version_id' \
        ' inner join program on program.id = program_version.program_id' \
        ' inner join platform on platform.id = platform_id' \
        ' inner join test on test.id = test_id' \
        ' where test.name = "pipe_zcu102" and tuning_run.name like "bnn_4x1.1x2.1x2_%%" and fidelity = 3'
data = pd.read_sql_query(query, engine)

# Ignore failed builds.
data = data[data['state'] == 'OK']
data = data.drop(columns=['state'])

# Convert the latency to seconds.
data['run_time'] /= data['proc_freq']
data = data.drop(columns=['proc_freq'])

# Compute the tuning time in days.
data['time'] = (data['collection_date'] - data['start_date']).dt.total_seconds() / 86400.0
data = data.drop(columns=['start_date', 'collection_date'])

# Truncate the results after 3 days.
data = data[data['time'] <= max_time]
data = data.drop(columns=['time'])

# Select the best configuration.
data = data.loc[data['run_time'].idxmin()]

# Get the best runtime.
best_run_time = data['run_time']

# Show the ID of the best configuration such that it can be used for an impact analysis run.
print('ID of best configuration:', data['id'])

# Show the runtime of the best configuration.
print('Runtime of best configuration:', best_run_time, 's\n')

# Extract the configuration.
base_cfg = loads(zlib.decompress(data['data']), encoding='latin1')

query = 'select data, result.state, run_time, proc_freq from result' \
        ' inner join configuration on result.configuration_id = configuration.id' \
        ' inner join tuning_run on tuning_run.id = result.tuning_run_id' \
        ' inner join program_version on program_version.id = program_version_id' \
        ' inner join program on program.id = program_version.program_id' \
        ' inner join platform on platform.id = platform_id' \
        ' inner join test on test.id = test_id' \
        ' where test.name = "sens_anal"'
data = pd.read_sql_query(query, engine)

cfgs = data['data'].transform(zlib.decompress)
data['cfg'] = cfgs.transform(lambda field: loads(field, encoding='latin1'))

# Convert the latency to seconds.
data['run_time'] /= data['proc_freq']
data = data.drop(columns=['proc_freq'])

# Assign runtimes to parameters that changed.
run_times = {}
for row in data.itertuples(index=False):
    for param, value in row.cfg.items():
        if value != base_cfg[param]:
            run_time = row.run_time if row.state == 'OK' else float('inf')
            run_times.setdefault(param, []).append(run_time)
            break

# Compute properties such as average, maximum, and success rate for each parameter.
summary = []
for param, values in run_times.items():
    valid_values = np.array(values)[~np.isinf(values)]
    success_rate = len(valid_values) / len(values)
    if len(valid_values) > 0:
        avg_value = np.mean(valid_values)
        std_dev = np.std(valid_values)
        max_value = np.max(valid_values)
    else:
        avg_value = float('inf')
        std_dev = float('inf')
        max_value = float('inf')
    if isinstance(base_cfg[param], float):
        best_value = '%.2f' % base_cfg[param]
    else:
        best_value = str(base_cfg[param])
    summary.append({'param': param,
                    'best_value': best_value,
                    'success_rate': success_rate,
                    'avg': avg_value,
                    'std_dev': std_dev,
                    'max': max_value})

# Put the properties back in a dataframe.
data = pd.DataFrame(data=summary)

# Order the parameters by success rate.
data = data.sort_values(['success_rate', 'max'], ascending=[True, False])

# We cannot show all parameters.
data = data.head(12)

# Generate Latex table.
output = ''
for id, row in data.iterrows():
    try:
        best_value = '%.0f' % float(row['best_value'])
    except:
        best_value = row['best_value']
    output += '\\texttt{' + row['param'].replace('_', '\_') + '}'
    output += ' & ' + best_value
    output += ' & %.0f\\%%' % (100.0 * row['success_rate'])
    if row['success_rate'] == 0:
        output += ' & - & -'
    else:
        output += ' & %.0f\\%%' % (100.0 * row['avg'] / best_run_time)
        output += ' & %.0f\\%%' % (100.0 * row['std_dev'] / best_run_time)
    output += ' \\\\\n'

# Show table.
print(output)

# Output table to file.
with open('sens_anal.tex', 'w') as output_file:
    output_file.write(output)

# Write the callouts to a file.
with open('../callouts/sens_anal.tex', 'w') as output_file:
    output_file.write('\\def \\bestruntime {%.0f}\n' % (1000.0 * best_run_time))
