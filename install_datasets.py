#!/usr/bin/env python3

import os
import subprocess
import yaml

script_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(script_dir, 'cfg.yml')) as input_file:
    data = input_file.read()
cfg = yaml.safe_load(data)

for platform in cfg['platforms']['instances']:
    print('Intalling datasets on %s...' % platform['hostname'])
    srcs = ['rosetta/bnn/data',
            'rosetta/bnn/params',
            'rosetta/optical-flow/datasets/current',
            'rosetta/spam-filter/data/shuffledfeats.dat',
            'rosetta/spam-filter/data/shuffledlabels.dat']
    srcs = [os.path.join(script_dir, src) for src in srcs]
    dst = platform['hostname'] + ':/run/media/mmcblk0p1'
    subprocess.run(['scp', '-r'] + srcs + [dst], check=True)
