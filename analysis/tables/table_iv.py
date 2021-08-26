#!/usr/bin/env python3

import math
import numpy as np
import os
import pandas as pd
import sqlalchemy
import yaml
from gzip import zlib
from pickle import loads

resources = ['luts', 'regs', 'dsps', 'brams']

platforms = {'zcu102': 'ZCU102', 'ultra96': 'Ultra96-V2', 'pynq': 'Pynq-Z1'}
benchmarks = {'bnn': 'BNN', 'spam-filter': 'Spam filter', '3d-rendering': '3D rendering',
              'digit-recognition': 'Digit recognition', 'optical-flow': 'Optical flow',
              'face-detection': 'Face detection'}


def get_value(data, benchmark, platform, technique, column):
    sel = data.loc[(data['benchmark'] == benchmark) &
                   (data['platform'] == platform) &
                   (data['technique'] == technique), column]
    return sel.squeeze() if len(sel) > 0 else float('inf')

def get_core_map(data, benchmark, platform):
    core_map = get_value(data, benchmark, platform, 'PipelinedMultiFidBayes', 'core_map')
    return core_map if core_map != 'bnn' else '4x1.1x2.1x2'

def get_runtime(data, benchmark, platform, technique):
    run_time = 1000.0 * get_value(data, benchmark, platform, technique, 'run_time')
    return '-' if math.isinf(run_time) else ('%.1f' % run_time)

def get_speedup(data, benchmark, platform, technique):
    run_time_untuned = untuned_latencies[(platform, benchmark)]
    run_time_bayes = 1000.0 * get_value(data, benchmark, platform, technique, 'run_time')
    return run_time_untuned / run_time_bayes

def get_utilization(data, benchmark, platform, technique, resource):
    usage = get_value(data, benchmark, platform, technique, resource)
    max_usage = get_value(data, benchmark, platform, technique, 'max_' + resource)
    utilization = 100.0 * usage / max_usage
    return '%.0f\\%%' % utilization

def get_untuned_latency(platform, benchmark):
    sel = data.loc[untuned_data['platform'] == platform & untuned_data['benchmark'] == benchmark, 'runtime']
    return ('%.1f' % (1000.0 * sel.squeeze())) if len(sel) > 0 else '-'

script_dir = os.path.dirname(os.path.realpath("__file__"))
with open('../../cfg.yml', 'r') as cfg_file:
    data = cfg_file.read()
tuner_cfg = yaml.safe_load(data)
database = tuner_cfg['database'].replace('mysql', 'mysql+pymysql')

engine = sqlalchemy.create_engine(database)
query = 'select platform.name as platform, program.name as benchmark, run_time, proc_freq from result' \
        ' inner join tuning_run on tuning_run.id = result.tuning_run_id' \
        ' inner join program_version on program_version.id = program_version_id' \
        ' inner join program on program.id = program_version.program_id' \
        ' inner join platform on platform.id = platform_id' \
        ' inner join test on test.id = test_id' \
        ' where test.name like "untuned_*"'
untuned_data = pd.read_sql_query(query, engine)

# Convert the latency to seconds.
untuned_data['run_time'] /= untuned_data['proc_freq']
untuned_data = untuned_data.drop(columns=['proc_freq'])

query = 'select program.name as benchmark, seed, args, proc_freq, tuning_run.start_date, collection_date,' \
        ' result.state, run_time, tuning_run.name, result.luts, result.regs, result.dsps, result.brams,' \
        ' platform.luts as max_luts, platform.regs as max_regs, platform.dsps as max_dsps,' \
        ' platform.brams as max_brams, platform.name as platform, fidelity from result' \
        ' inner join configuration on result.configuration_id = configuration.id' \
        ' inner join tuning_run on tuning_run.id = result.tuning_run_id' \
        ' inner join program_version on program_version.id = program_version_id' \
        ' inner join program on program.id = program_version.program_id' \
        ' inner join platform on platform.id = platform_id' \
        ' inner join test on test.id = test_id' \
        ' where test.name like "opentuner_%%" or test.name like "bayes_%%" or' \
        '       test.name like "pipe_%%" and (tuning_run.name not like "bnn_%%" or' \
        '                                     tuning_run.name like "bnn_4x1.1x2.1x2_%%")'
data = pd.read_sql_query(query, engine)

# Get command line arguments of each tuning run.
args = data['args'].transform(zlib.decompress)
args = args.transform(lambda field: loads(field, encoding='latin1'))
data = data.drop(columns=['args'])

# Get the search technique of each tuning run.
data['technique'] = args.transform(lambda field: field.technique[0])

# Extract the core assignment of each tuning_run.
data['core_map'] = args.transform(lambda field: (field.core_map if 'core_map' in field else ''))
data.loc[data['core_map'].isna(), 'core_map'] = ''

# Determine the maximum fidelity.
data.loc[data['technique'] == 'PipelinedMultiFidBayes', 'max_fidelity'] = 3
data.loc[data['technique'] == 'AUCBanditMetaTechniqueA', 'max_fidelity'] = 0
data.loc[data['technique'] == 'Bayes', 'max_fidelity'] = 0

# Extract the seed from the tuning run name.
data['seed'] = data['name'].str.extract(r'.*_(\d+)$').astype(int)

# Replace extremely large values with infinity.
data.loc[data['run_time'] > 1e30, 'run_time'] = float('inf')

# Set runtime of failed builds to infinity to make sure that they will be ignored later.
data.loc[data['state'] != 'OK', 'run_time'] = float('inf')
data = data.drop(columns=['state'])

# Set runtime of incomplete builds to infinity to make sure that they will be ignored later.
data.loc[data['fidelity'] != data['max_fidelity'], 'run_time'] = float('inf')
data = data.drop(columns=['fidelity', 'max_fidelity'])

# Set the resource usage of all failed or incomplete builds to NA.
data.loc[~np.isfinite(data['run_time']), resources] = None

# Convert the latency to seconds.
data['run_time'] = data['run_time'] / data['proc_freq']
data = data.drop(columns=['proc_freq'])

# Compute the tuning time in days.
data['time'] = (data['collection_date'] - data['start_date']).dt.total_seconds() / 86400.0
data = data.drop(columns=['start_date', 'collection_date'])

# Cut the tuning runs off after 1 day.
data_1 = data[data['time'] < 1]

# Find the shortest latency.  We sort in case there are ties.
data_1 = data_1.sort_values('time')
data_1 = data_1.drop(columns=['time'])
rows = data_1.groupby(['benchmark', 'platform', 'seed', 'technique', 'core_map'])['run_time'].idxmin()
data_1 = data_1.loc[rows]

# Find average latency across seeds.
data_1 = data_1.groupby(['benchmark', 'platform', 'technique', 'core_map']).mean().reset_index()
data_1 = data_1.drop(columns=['seed'])

# Cut the tuning runs off after 3 days.
data = data[data['time'] < 3]
data = data.drop(columns=['time'])

# Find the shortest latency.  We sort in case there are ties.
rows = data.groupby(['benchmark', 'platform', 'seed', 'technique', 'core_map'])['run_time'].idxmin()
data = data.loc[rows]

# Find average latency across seeds.
data = data.groupby(['benchmark', 'platform', 'technique', 'core_map']).mean().reset_index()
data = data.drop(columns=['seed'])

# Generate Latex table.
output = ''
min_speedup_1 = float('inf')
min_speedup_3 = float('inf')
max_speedup_1 = 0.0
max_speedup_3 = 0.0
speedups_bayes = []
speedups_pipe = []
for (benchmark, platform), group in data.groupby(['benchmark', 'platform']):
    core_map = get_core_map(data, benchmark, platform)
    core_map = '\\texttt{' + core_map.replace('.', ',') + '}'
    speedup_bayes = get_speedup(data, benchmark, platform, 'Bayes')
    speedup_1 = get_speedup(data_1, benchmark, platform, 'PipelinedMultiFidBayes')
    speedup_3 = get_speedup(data, benchmark, platform, 'PipelinedMultiFidBayes')
    output += benchmarks[benchmark] + ' & ' + platforms[platform] + ' & ' + core_map.replace('.', ',')
    output += ' & ' + get_untuned_latency(platform, benchmark)
    output += ' & ' + get_runtime(data_1, benchmark, platform, 'AUCBanditMetaTechniqueA')
    output += ' & ' + get_runtime(data, benchmark, platform, 'AUCBanditMetaTechniqueA')
    output += ' & ' + get_runtime(data_1, benchmark, platform, 'Bayes')
    output += ' & ' + get_runtime(data, benchmark, platform, 'Bayes')
    output += ' & ' + get_runtime(data_1, benchmark, platform, 'PipelinedMultiFidBayes')
    output += ' & ' + get_runtime(data, benchmark, platform, 'PipelinedMultiFidBayes')
    output += ' & ' + get_utilization(data, benchmark, platform, 'PipelinedMultiFidBayes', 'luts')
    output += ' & ' + get_utilization(data, benchmark, platform, 'PipelinedMultiFidBayes', 'regs')
    output += ' & ' + get_utilization(data, benchmark, platform, 'PipelinedMultiFidBayes', 'dsps')
    output += ' & ' + get_utilization(data, benchmark, platform, 'PipelinedMultiFidBayes', 'brams') + ' \\\\\n'
    if platform == 'zcu102':
        min_speedup_1 = min(min_speedup_1, speedup_1)
        min_speedup_3 = min(min_speedup_3, speedup_3)
        max_speedup_1 = max(max_speedup_1, speedup_1)
        max_speedup_3 = max(max_speedup_3, speedup_3)
        speedups_bayes.append(speedup_bayes)
        speedups_pipe.append(speedup_3)

# Show table.
print(output)

# Output table to file.
with open('run_time.tex', 'w') as output_file:
    output_file.write(output)

# Compute the average runtime decrease after 1 day.
sel = data_1[data_1['platform'] == 'zcu102'].sort_values('benchmark')
data_opentuner = sel.loc[sel['technique'] == 'AUCBanditMetaTechniqueA', 'run_time'].reset_index(drop=True)
data_pipe = sel.loc[sel['technique'] == 'PipelinedMultiFidBayes', 'run_time'].reset_index(drop=True)
run_time_dec = 100.0 * (1.0 - (data_pipe / data_opentuner).mean())

# Compute the average runtime decrease after 3 days.
sel = data[data['platform'] == 'zcu102'].sort_values('benchmark')
data_opentuner = sel.loc[sel['technique'] == 'AUCBanditMetaTechniqueA', 'run_time'].reset_index(drop=True)
data_bayes = sel.loc[sel['technique'] == 'Bayes', 'run_time'].reset_index(drop=True)
data_pipe = sel.loc[sel['technique'] == 'PipelinedMultiFidBayes', 'run_time'].reset_index(drop=True)
run_time_dec_opent = 100.0 * (1.0 - (data_pipe / data_opentuner).mean())
run_time_dec_bayes = 100.0 * (1.0 - (data_pipe / data_bayes).mean())

# Compute the speedup decrease due to not using the multi-fidelity model.
speedup_dec = 100.0 * (1.0 - np.mean(speedups_bayes) / np.mean(speedups_pipe))

# Show the runtime decrease from using the pipeline vs OpenTuner.
print('Runtime decrease with the pipeline vs OpenTuner after 1 day: %f%%' % run_time_dec)
print('Runtime decrease with the pipeline vs Bayesian optimization after 3 days: %f%%' % run_time_dec_bayes)
print('Runtime decrease with the pipeline vs OpenTuner after 3 days: %f%%' % run_time_dec_opent)

# Show the speedups.
print('Speedup after 1 day: %f - %f' % (min_speedup_1, max_speedup_1))
print('Speedup after 3 days: %f - %f' % (min_speedup_3, max_speedup_3))

# Show the speedup decreases.
print('Speedup decrease: %f%%' % speedup_dec)
                
# Output callouts to file.
with open('../callouts/run_time.tex', 'w') as output_file:
    output_file.write('\\def \\runtimedec {%.0f}\n' % run_time_dec)
    output_file.write('\\def \\runtimedecbayes {%.0f}\n' % run_time_dec_bayes)
    output_file.write('\\def \\runtimedecopent {%.0f}\n' % run_time_dec_opent)
    output_file.write('\\def \\minspeedupone {{{:0.1f}}}\n'.format(min_speedup_1))
    output_file.write('\\def \\maxspeedupone {{{:0.0f}}}\n'.format(max_speedup_1))
    output_file.write('\\def \\minspeedupthree {{{:0.1f}}}\n'.format(min_speedup_3))
    output_file.write('\\def \\maxspeedupthree {{{:0.0f}}}\n'.format(max_speedup_3))
    output_file.write('\\def \\bayesspeedupdec {%.0f}\n' % speedup_dec)                
