#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sqlalchemy
import yaml
from gzip import zlib
from pickle import loads

# Note that this function cannot handle multiple rows from the same timestamp.
def resample_seed(times, data):
    time_step = 0
    rows = []
    infinite_rows = []
    for row, time in enumerate(times):
        while time_step < len(data):
            if data['time'].iat[time_step] > time:
                break
            time_step += 1
        if time_step > 0:
            rows.append(time_step - 1)
        else:
            rows.append(0)
            infinite_rows.append(row)
    output = data.iloc[rows].reset_index(drop=True)
    output['run_time'].iloc[infinite_rows] = float('inf')
    output['time'] = times
    return output

def resample(data):
    data = data.sort_values('time')
    end_time = data.groupby(['seed'])['time'].max().min()
    times = data['time'].unique()
    times = times[times <= end_time]
    data = data.groupby(['seed']).apply(lambda group: resample_seed(times, group))
    return data

def mean_std(values):
    std_dev = np.std(values) / np.sqrt(len(values))
    return float('inf') if np.isnan(std_dev) else std_dev

script_dir = os.path.dirname(os.path.realpath("__file__"))
with open('../../cfg.yml', 'r') as cfg_file:
    data = cfg_file.read()
tuner_cfg = yaml.safe_load(data)
database = tuner_cfg['database'].replace('mysql', 'mysql+pymysql')

engine = sqlalchemy.create_engine(database)
query = 'select program.name as benchmark, platform.name as platform, seed, args, proc_freq, test.name,' \
        ' tuning_run.start_date, collection_date, fidelity, result.state, run_time from result' \
        ' inner join configuration on result.configuration_id = configuration.id' \
        ' inner join desired_result on desired_result.result_id = result.id' \
        ' inner join tuning_run on tuning_run.id = result.tuning_run_id' \
        ' inner join program_version on program_version.id = program_version_id' \
        ' inner join program on program.id = program_version.program_id' \
        ' inner join platform on platform.id = platform_id' \
        ' inner join test on test.id = test_id' \
        ' where test.name = "no_sampling" or' \
        '       test.name = "bayes_zcu102" and tuning_run.name like "bnn_%%"'
data = pd.read_sql_query(query, engine)

# Get command line arguments of each tuning run.
args = data['args'].transform(zlib.decompress)
args = args.transform(lambda field: loads(field, encoding='latin1'))
data = data.drop(columns=['args'])

# Get the search technique of each tuning run.
data['technique'] = args.transform(lambda field: field.technique[0])

# Set fidelity of all complete builds to 1.
data.loc[data['fidelity'].isna(), 'fidelity'] = 1

# Determine the maximum fidelity.
data.loc[data['technique'] == 'Bayes', 'max_fidelity'] = 0
data.loc[data['technique'] == 'MultiFidBayes', 'max_fidelity'] = 3
data.loc[data['technique'] == 'AUCBanditMetaTechniqueA', 'max_fidelity'] = 1
data.loc[data['technique'] == 'Random', 'max_fidelity'] = 1

# Replace extremely large values with infinity.
data.loc[data['run_time'] > 1e30, 'run_time'] = float('inf')

# Set runtime of failed builds to infinity to make sure that they will be ignored later.
data.loc[data['state'] != 'OK', 'run_time'] = float('inf')
data = data.drop(columns=['state'])

# Set runtime of incomplete builds to infinity to make sure that they will be ignored later.
data.loc[data['fidelity'] != data['max_fidelity'], 'run_time'] = float('inf')
data = data.drop(columns=['fidelity', 'max_fidelity'])

# Convert the latency to seconds.
data['run_time'] = data['run_time'] / data['proc_freq']
data = data.drop(columns=['proc_freq'])

# Compute the tuning time in days.
data['time'] = (data['collection_date'] - data['start_date']).dt.total_seconds() / 86400.0
data = data.drop(columns=['start_date', 'collection_date'])

# Order the data by time.
data = data.sort_values('time')

# Use the smallest value if multiple measurements have the same timestamp.
data = data.loc[data.groupby(['name', 'seed', 'time'])['run_time'].idxmin()]

# Compute the shortest runtime so far.
data['run_time'] = data.groupby(['name', 'seed'])['run_time'].cummin()

# Use a common time series for each experiment with the same seeds.
data = data.groupby('name').apply(resample).reset_index(drop=True)

# Compute the average and standard deviation of the average.
groups = data.groupby(['benchmark', 'platform', 'name', 'time'])
data = groups.agg(mean=('run_time', 'mean'), std=('run_time', mean_std), seeds=('seed', 'nunique')).reset_index()

# Compute the top and bottom of the uncertainty regions.
data['min'] = data['mean'] - data['std']
data['max'] = data['mean'] + data['std']

# Increase the font size.
plt.rcParams.update({'font.size': 15})

platforms = {'zcu102': 'ZCU102', 'ultra96': 'Ultra96-V2', 'pynq': 'Pynq-Z1'}
benchmarks = {'bnn': 'BNN', 'spam-filter': 'Spam filter', '3d-rendering': '3D rendering',
              'digit-recognition': 'Digit recognition', 'optical-flow': 'Optical flow',
              'face-detection': 'Face detection'}
descriptions = {'no_samp': 'Numerical optimization', 'bayes_zcu102': 'Random sampling'}
for (benchmark, platform), group1 in data.groupby(['benchmark', 'platform']):
    plots = []
    names = []
    for (name, seeds), group2 in group1.groupby(['name', 'seeds']):
        plot_line = plt.step(group2['time'], group2['mean'], where='post')[0]
        plot_fill = plt.fill_between(group2['time'], group2['min'], group2['max'], step='post', alpha=0.2)
        plots.append((plot_line, plot_fill))
        names.append(descriptions[name])
    plt.xlim([0, 3.0])
    #plt.ylim([0, None])
    plt.ylim([0.08, 2.0])
    plt.yscale('log')
    plt.title('{} on {} ({} repetitions)'.format(benchmarks[benchmark], platforms[platform], seeds))
    plt.xlabel('Tuning time (days)')
    plt.ylabel('Runtime (seconds)')
    plt.legend(plots, names)
    plt.savefig('figure_5.pdf', bbox_inches='tight')
