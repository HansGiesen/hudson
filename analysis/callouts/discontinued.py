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
query = 'select fidelity, result.state, msg, configuration.hash from result' \
        ' inner join configuration on result.configuration_id = configuration.id' \
        ' inner join tuning_run on tuning_run.id = result.tuning_run_id' \
        ' inner join program_version on program_version.id = program_version_id' \
        ' inner join program on program.id = program_version.program_id' \
        ' inner join test on test.id = test_id' \
        ' where test.name = "pipe_zcu102" and tuning_run.name like "bnn_4x1.1x2.1x2_%%"'
data = pd.read_sql_query(query, engine)

# Get rid of rows that are not at the highest fidelity.
data = data.loc[data.groupby('hash')['fidelity'].idxmax()]
data = data.drop(columns=['hash'])

# Count the number of configurations that completed presynthesis, but did not continue.
presynth_ok = ((data['fidelity'] == 1) & (data['state'] == 'OK')).sum()

# Count the number of configurations that completed presynthesis, but did not continue.
synth_ok = ((data['fidelity'] == 2) & (data['state'] == 'OK')).sum()

# Count the total number of configurations.
cfg_cnt = len(data)

# Show the percentage of configurations completing presynthesis, but not continuing.
presynth_term_perc = presynth_ok / cfg_cnt * 100.0
print('Discontinued presynthesis percentage: {}%'.format(presynth_term_perc))

# Show the percentage of configurations completing presynthesis, but not continuing.
synth_term_perc = synth_ok / cfg_cnt * 100.0
print('Discontinued synthesis percentage: {}%'.format(synth_term_perc))

# Output percentages to file.
with open('../callouts/discontinued.tex', 'w') as output_file:
    output_file.write('\\def \\presynthtermperc {{{:0.1f}\%}}\n'.format(presynth_term_perc))
    output_file.write('\\def \\synthtermperc {{{:0.1f}\%}}\n'.format(synth_term_perc))
