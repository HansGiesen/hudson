#!/usr/bin/env python3

import glob
import numpy as np
import os
import pandas as pd
import re
import sqlalchemy
import sys
import yaml
from gzip import zlib
from pickle import loads

ref_time = 1.0

techniques = {'AUCBanditMetaTechniqueA': 'OpenTuner', 'PipelinedMultiFidBayes': 'HuDSoN'}

def extract_parallelism(field):
    if hasattr(field, 'core_map'):
        return int(re.match(r'(\d+)x', field.core_map).group(1))
    else:
        return field.parallelism

def compute_success_prob(data):
    return sum(data['run_time'] < 1e30) / data['hash'].nunique()

def compute_avg_search_time(data):
    search_times = data.groupby('hash')['search_time'].sum()
    return search_times.mean()

def compute_avg_build_time(data):
    build_times = data.loc[data['fidelity'] <= 1, 'build_time']
    build_times = build_times[build_times < 1e30]
    return build_times.mean()

pd.set_option('display.max_rows', None)

script_dir = os.path.dirname(os.path.realpath("__file__"))
with open('../../cfg.yml', 'r') as cfg_file:
    data = cfg_file.read()
tuner_cfg = yaml.safe_load(data)
database = tuner_cfg['database'].replace('mysql', 'mysql+pymysql')

engine = sqlalchemy.create_engine(database)
query = 'select seed, args, proc_freq, tuning_run.start_date, collection_date, search_time, build_time,' \
        ' result.state, run_time, test.name as test, tuning_run.name, fidelity, configuration.hash from result' \
        ' inner join desired_result on desired_result.result_id = result.id' \
        ' inner join configuration on result.configuration_id = configuration.id' \
        ' inner join tuning_run on tuning_run.id = result.tuning_run_id' \
        ' inner join program_version on program_version.id = program_version_id' \
        ' inner join program on program.id = program_version.program_id' \
        ' inner join platform on platform.id = platform_id' \
        ' inner join test on test.id = test_id' \
        ' where (test.name = "opentuner_zcu102" and program.name = "bnn") or' \
        '       (test.name = "pipe_zcu102" and tuning_run.name like "bnn_4x1.1x2.1x2_%%")'
data = pd.read_sql_query(query, engine)

# I accidentally forgot to set the seed for some OpenTuner runs, but I can extract it from the name.
data['seed'] = data['name'].str.extract(r'.*_(\d+)$').astype(int)

# Extract the names of the tuning runs.
data['name'] = data['name'].str.extract(r'(.*)_\d+$')

# Get command line arguments of each tuning run.
args = data['args'].transform(zlib.decompress)
args = args.transform(lambda field: loads(field, encoding='latin1'))
data = data.drop(columns=['args'])

# Get the search technique of each tuning run.
data['technique'] = args.transform(lambda field: field.technique[0])

# Get the amount of parallelism.
data['parallelism'] = args.transform(extract_parallelism)

# Determine the maximum fidelity.
data.loc[data['technique'] == 'PipelinedMultiFidBayes', 'max_fidelity'] = 3
data.loc[data['technique'] == 'AUCBanditMetaTechniqueA', 'max_fidelity'] = 0

# Replace extremely large values with infinity.
data.loc[data['run_time'] > 1e30, 'run_time'] = float('inf')

# Set runtime of failed builds to infinity to make sure that they will be ignored later.
data.loc[data['state'] != 'OK', 'run_time'] = float('inf')

# Set runtime of incomplete builds to infinity to make sure that they will be ignored later.
data.loc[data['fidelity'] != data['max_fidelity'], 'run_time'] = float('inf')
data = data.drop(columns=['max_fidelity'])

# Convert the latency to seconds.
data['run_time'] = data['run_time'] / data['proc_freq']
data = data.drop(columns=['proc_freq'])

# Compute the tuning time in days.
data['time'] = (data['collection_date'] - data['start_date']).dt.total_seconds()
data = data.drop(columns=['start_date', 'collection_date'])

# Get the OpenTuner results.
data_opent = data[data['technique'] == 'AUCBanditMetaTechniqueA']

# Determine the best runtime when the tuning time is limited to the reference time.
data_opent = data_opent[data_opent['time'] < ref_time * 86400.0]
ref_run_times = data_opent.groupby(['seed'])['run_time'].min()

# Find the average runtime over all seeds.
ref_run_time = ref_run_times.mean()
ref_run_time = 0.2

# Show the reference runtime.
print('Reference runtime:', ref_run_time, 's')

# Determine how long it takes for each method to reach the reference runtime.
sel = data[data['run_time'] <= ref_run_time]
end_times = sel.groupby(['technique', 'seed']).min()
columns = ['run_time', 'search_time', 'build_time', 'parallelism', 'test', 'state', 'name', 'fidelity', 'hash']
end_times = end_times.drop(columns=columns)

# Merge the end times back into the results.
end_times = end_times.rename(columns={'time': 'end_time'})
data = data.set_index(['technique', 'seed']).join(end_times)

# Remove all data before the end time.
data = data[data['time'] <= data['end_time']]

search_times = {}
full_build_cnts = {}
overhead_times = {}
tuning_times = {}
for technique, group in data.groupby('technique'):
    
    test = group.iloc[0]['test']
    tuning_run = group.iloc[0]['name']
    script_dir = os.path.dirname(os.path.realpath("__file__"))
    test_dir = os.path.join(script_dir, '../../test', test)
    
    avg_eval_times = []
    for filename in glob.iglob(os.path.join(test_dir, tuning_run + '_*/stderr.log')):
        eval_times = []
        with open(filename, 'r') as log_file:
            for line in log_file:
                match = re.match(r'\[\s*(\d+)s.*Build.*Running on', line)
                if match:
                    start = int(match.group(1))
                match = re.match(r'\[\s*(\d+)s.*Run was successful.', line)
                if match:
                    end = int(match.group(1))
                    eval_times.append(end - start)
                match = re.match(r'\[\s*(\d+)s.*Run did not pass', line)
                if match:
                    end = int(match.group(1))
                    eval_times.append(end - start)
        avg_eval_times.append(sum(eval_times) / len(eval_times))
    avg_eval_time = sum(avg_eval_times) / len(avg_eval_times)
    
    cfg_cnt = group.groupby('seed')['hash'].nunique().mean()
    success_prob = group.groupby('seed').apply(compute_success_prob).mean()
    avg_search_time = group.groupby('seed').apply(compute_avg_search_time).mean()
    avg_build_time = group.groupby('seed').apply(compute_avg_build_time).mean()
    batch_size = group.iloc[0]['parallelism']
    avg_busy_time = avg_search_time + avg_build_time / batch_size + success_prob * avg_eval_time
    avg_tuning_time = group.groupby('seed')['end_time'].mean().mean()
    overhead_time = avg_tuning_time / cfg_cnt - avg_busy_time

    full_build_cnts[technique] = cfg_cnt * success_prob
    search_times[technique] = avg_search_time
    overhead_times[technique] = overhead_time
    tuning_times[technique] = avg_tuning_time
    
    output = ('%.1f & %d & %.2f & %.2f & %.0f & %.0f & %.0f & %.0f \\\\' %
             (cfg_cnt, batch_size, success_prob, avg_search_time, avg_build_time, avg_eval_time, overhead_time,
              avg_tuning_time))

    # Show table.
    print(techniques[technique] + ':', output)
    
    # Output table to file.
    suffix = 'bayes' if technique == 'PipelinedMultiFidBayes' else 'opentuner'
    with open('model_params_%s.tex' % suffix, 'w') as output_file:
        output_file.write(output + '\n')
        
    # Compute the percentage of time spent on building.
    if technique == 'AUCBanditMetaTechniqueA':
        build_time_perc = (avg_build_time / batch_size) / avg_busy_time * 100.0
        print('Build time percentage:', build_time_perc)

# Compute the ratios between the search times and overhead.
search_time_ratio = search_times['PipelinedMultiFidBayes'] / search_times['AUCBanditMetaTechniqueA']
overhead_time_ratio = overhead_times['AUCBanditMetaTechniqueA'] / overhead_times['PipelinedMultiFidBayes']

# Compute the tuning time reduction.
tuning_time_dec = 100.0 * (1.0 - tuning_times['PipelinedMultiFidBayes'] / tuning_times['AUCBanditMetaTechniqueA'])
        
# Show the ratios.
print('Full builds with OpenTuner:', full_build_cnts['AUCBanditMetaTechniqueA'])
print('Full builds with Bayesian optimization:', full_build_cnts['PipelinedMultiFidBayes'])
print('Ratio between search times:', search_time_ratio)
print('Ratio between overhead times:', overhead_time_ratio)
print('Tuning time decrease: %f%%' % tuning_time_dec)
    
# Write the callouts to a file.
with open('../callouts/model_params.tex', 'w') as output_file:
    output_file.write('\\def \\tuningtimethres {%.0f}\n' % (ref_run_time * 1000.0))
    output_file.write('\\def \\buildtimeperc {%.1f}\n' % build_time_perc)
    output_file.write('\\def \\fullbuildsopent {%.1f}\n' % full_build_cnts['AUCBanditMetaTechniqueA'])
    output_file.write('\\def \\fullbuildsbayes {%.1f}\n' % full_build_cnts['PipelinedMultiFidBayes'])
    output_file.write('\\def \\searchtimeratio {%.0f}\n' % search_time_ratio)
    output_file.write('\\def \\overheadratio {%.0f}\n' % overhead_time_ratio)
    output_file.write('\\def \\tuningtimedec {%.0f}\n' % tuning_time_dec)
