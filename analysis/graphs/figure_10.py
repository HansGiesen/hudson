#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import sqlalchemy
import yaml
from gzip import zlib
from pickle import loads

ref_time = 3.0

benchmarks = {'bnn': 'BNN', 'spam-filter': 'Spam filter', '3d-rendering': '3D rendering',
              'digit-recognition': 'Digit recognition', 'optical-flow': 'Optical flow',
              'face-detection': 'Face detection'}

pd.set_option('display.max_rows', None)

def add_worst_case(data):
    row = data.iloc[-1]
    row['time'] = float('inf')
    row['run_time'] = 0.0
    return data.append(row)

script_dir = os.path.dirname(os.path.realpath("__file__"))
with open('../../cfg.yml', 'r') as cfg_file:
    data = cfg_file.read()
tuner_cfg = yaml.safe_load(data)
database = tuner_cfg['database'].replace('mysql', 'mysql+pymysql')

engine = sqlalchemy.create_engine(database)
query = 'select program.name as benchmark, args, proc_freq, tuning_run.start_date, collection_date,' \
        ' result.state, run_time, fidelity, tuning_run.name from result' \
        ' inner join configuration on result.configuration_id = configuration.id' \
        ' inner join tuning_run on tuning_run.id = result.tuning_run_id' \
        ' inner join program_version on program_version.id = program_version_id' \
        ' inner join program on program.id = program_version.program_id' \
        ' inner join platform on platform.id = platform_id' \
        ' inner join test on test.id = test_id' \
        ' where test.name = "opentuner_zcu102" or test.name = "bayes_zcu102" or ' \
        '       test.name = "pipe_zcu102" and tuning_run.name like "bnn_4x1.1x2.1x2_%%"'
data = pd.read_sql_query(query, engine)

# I accidentally forgot to set the seed for some OpenTuner runs, but I can extract it from the name.
data['seed'] = data['name'].str.extract(r'.*_(\d+)$').astype(int)
data = data.drop(columns=['name'])

# Get command line arguments of each tuning run.
args = data['args'].transform(zlib.decompress)
args = args.transform(lambda field: loads(field, encoding='latin1'))
data = data.drop(columns=['args'])

# Get the search technique of each tuning run.
data['technique'] = args.transform(lambda field: field.technique[0])

# Determine the maximum fidelity.
data.loc[data['technique'] == 'PipelinedMultiFidBayes', 'max_fidelity'] = 3
data.loc[data['technique'] == 'AUCBanditMetaTechniqueA', 'max_fidelity'] = 0
data.loc[data['technique'] == 'Bayes', 'max_fidelity'] = 0

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

# Add a worst-case runtime to each experiment
data = data.groupby(['technique', 'benchmark', 'seed']).apply(add_worst_case).reset_index(drop=True)


# Compute the threshold.

# Extract OpenTuner results.
data_opentuner = data[data['technique'] == 'AUCBanditMetaTechniqueA']
data_opentuner = data_opentuner.drop(columns=['technique'])

# Cut the OpenTuner tuning runs off after the specified number of days.
thresholds = data_opentuner[data_opentuner['time'] < ref_time]

# Compute the minimum latency.  We sort in case there are ties.
thresholds = thresholds.sort_values('time')
thresholds = thresholds.groupby(['benchmark', 'seed']).min()
thresholds = thresholds.drop(columns=['time'])

# Average the minimum latency over the seeds to get the latency threshold.
thresholds = thresholds.groupby(['benchmark']).mean()

# Rename the runtime column.
thresholds = thresholds.rename(columns={'run_time': 'thres'})


# Determine when the threshold is reached in single-fidelity Bayesian optimization.

# Extract single-fidelity Bayesian optimization results.
data_bayes = data[data['technique'] == 'Bayes']
data_bayes = data_bayes.drop(columns=['technique'])

# Remove results that are worse than the threshold.
data_bayes = data_bayes.set_index('benchmark').join(thresholds).reset_index()
data_bayes = data_bayes[data_bayes['run_time'] <= data_bayes['thres']]
data_bayes = data_bayes.drop(columns=['run_time', 'thres'])

# Find the earliest time at which the threshold was not exceeded.
data_bayes = data_bayes.groupby(['benchmark', 'seed']).min().reset_index()


# Determine when the threshold is reached in Bayesian optimization pipeline.

# Extract Bayesian optimization pipeline results.
data_pipe = data[data['technique'] == 'PipelinedMultiFidBayes']
data_pipe = data_pipe.drop(columns=['technique'])

# Remove results that are worse than the threshold.
data_pipe = data_pipe.set_index('benchmark').join(thresholds).reset_index()
data_pipe = data_pipe[data_pipe['run_time'] <= data_pipe['thres']]
data_pipe = data_pipe.drop(columns=['run_time', 'thres'])

# Find the earliest time at which the threshold was not exceeded.
data_pipe = data_pipe.groupby(['benchmark', 'seed']).min().reset_index()


# Determine when the threshold is reached in OpenTuner.

# Average the run times over all seeds.
run_times = {}
new_data = {}
seed_cnt = data_opentuner['benchmark'].nunique()
for id, row in data_opentuner.sort_values('time').iterrows():
    benchmark = row['benchmark']
    seed = row['seed']
    time = row['time']
    if benchmark not in run_times:
        run_times[benchmark] = [float('inf')] * seed_cnt
    old_run_time = np.mean(run_times[benchmark])
    run_times[benchmark][seed] = min(run_times[benchmark][seed], row['run_time'])
    run_time = np.mean(run_times[benchmark])
    if run_time < old_run_time:
        new_data.setdefault(benchmark, {})[time] = run_time
data_opentuner = []
for benchmark, data in new_data.items():
    for time, run_time in data.items():
        data_opentuner.append({'benchmark': benchmark, 'time': time, 'run_time': run_time})
data_opentuner = pd.DataFrame(data_opentuner)

# Remove results that are worse than the threshold.
data_opentuner = data_opentuner.set_index('benchmark').join(thresholds).reset_index()
data_opentuner = data_opentuner[data_opentuner['run_time'] <= data_opentuner['thres']]
data_opentuner = data_opentuner.drop(columns=['run_time', 'thres'])

# Find the earliest time at which the threshold was not exceeded.
data_opentuner = data_opentuner.groupby(['benchmark']).min()


# Combine the data frames.
data = data_opentuner.join(data_bayes.set_index('benchmark'), lsuffix='_opentuner', rsuffix='_bayes')
data_pipe = data_pipe.rename(columns={'time': 'time_pipe'})
data = data.set_index('seed', append=True).join(data_pipe.set_index(['benchmark', 'seed']))

# Compute the speedups.
data['speedup_bayes'] = data['time_opentuner'] / data['time_bayes']
data['speedup_pipe'] = data['time_opentuner'] / data['time_pipe']

# Increase the font size.
plt.rcParams.update({'font.size': 15})

values = []
labels = []
for benchmark, group in data.groupby('benchmark'):
    for technique in ['bayes', 'pipe']:
        values.append(group['speedup_' + technique])
    labels.append(benchmarks[benchmark])
box = plt.boxplot(values, patch_artist=True, boxprops=dict(facecolor='white', color='black'))
seed_cnt = data.reset_index()['seed'].nunique()
plt.title('ZCU102, {} repetitions'.format(seed_cnt))
plt.ylabel('Speedup')
axes = plt.gca()
axes.set_yscale('log')
axes.xaxis.set_major_locator(ticker.FixedLocator([0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5]))
axes.xaxis.set_minor_locator(ticker.FixedLocator([1.5, 3.5, 5.5, 7.5, 9.5, 11.5]))
axes.xaxis.set_major_formatter(ticker.NullFormatter())
axes.xaxis.set_minor_formatter(ticker.FixedFormatter(labels))
for tick in axes.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment('right')
axes.tick_params(axis="x", which="minor", rotation=30)
plt.setp(box['boxes'][1::2], color='#C0FFC0')
plt.setp(box['boxes'][::2], color='#FFC0C0')
plt.setp(box['boxes'], edgecolor='black')
plt.setp(box['medians'], color='black')
plt.grid(axis='y')
plt.ylim([0.7, None])
plt.legend(box["boxes"][:2], ['SF BO', 'MF BO'])
plt.savefig('tuning_time.pdf', bbox_inches='tight')

# Compute the average speedups.
avg_speedup_bayes = data['speedup_bayes'].mean()
avg_speedup_pipe = data['speedup_pipe'].mean()

# Show the average speedup.
print('Average tuning time speedup without pipeline:', avg_speedup_bayes)
print('Average tuning time speedup with pipeline:', avg_speedup_pipe)

# Output percentages to file.
with open('../callouts/tuning_time.tex', 'w') as output_file:
    output_file.write('\\def \\avgspeedupbayes {{{:0.1f}}}\n'.format(avg_speedup_bayes))
    output_file.write('\\def \\avgspeeduppipe {{{:0.1f}}}\n'.format(avg_speedup_pipe))
