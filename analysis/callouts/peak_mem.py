#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import re
import sqlalchemy
import yaml
from gzip import zlib
from pickle import loads

parallelism = 4

script_dir = os.path.dirname(os.path.realpath("__file__"))
with open('../../cfg.yml', 'r') as cfg_file:
    data = cfg_file.read()
tuner_cfg = yaml.safe_load(data)
database = tuner_cfg['database'].replace('mysql', 'mysql+pymysql')

engine = sqlalchemy.create_engine(database)
query = 'select fidelity, mem_usage, args from result' \
        ' inner join configuration on configuration.id = result.configuration_id' \
        ' inner join tuning_run on tuning_run.id = result.tuning_run_id' \
        ' inner join program_version on program_version.id = program_version_id' \
        ' inner join program on program.id = program_version.program_id' \
        ' inner join test on test.id = test_id' \
        ' where test.name = "pipe_zcu102" and tuning_run.name like "bnn_4x1.1x2.1x2_%%"'
data = pd.read_sql_query(query, engine)

# Get command line arguments of each tuning run.
args = data['args'].transform(zlib.decompress)
args = args.transform(lambda field: loads(field, encoding='latin1'))
data = data.drop(columns=['args'])

# Get the core assignment.
core_map = args.iloc[0].core_map

# Extract the number of build processes at each fidelity.
procs = re.match(r'(\d+)x\d+\.(\d+)x\d+\.(\d+)x\d+', core_map).groups()

# Ignore results without memory usage information.
data = data[data['mem_usage'] < 1e30]

# Compute the maximum memory usage per fidelity.
mem_usages = data.groupby('fidelity')['mem_usage'].max()

peak_mem_opentuner = mem_usages.max() * parallelism
peak_mem_hudson = np.sum(mem_usages.to_numpy() * np.array([int(cnt) for cnt in procs]))

# Show the core utilization.
print('Peak memory consumption with OpenTuner:', peak_mem_opentuner)
print('Peak memory consumption with HuDSoN:', peak_mem_hudson)

# Output callouts to file.
with open('../callouts/peak_mem.tex', 'w') as output_file:
    output_file.write('\\def \\peakmemopentuner {%0.1f}\n' % peak_mem_opentuner)
    output_file.write('\\def \\peakmemhudson {%0.1f}\n' % peak_mem_hudson)
