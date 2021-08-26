#!/usr/bin/env python3

import argparse
import os
import yaml

script_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(script_dir, '../cfg.yml')) as input_file:
    data = input_file.read()
cfg = yaml.safe_load(data)

parser = argparse.ArgumentParser(description='Get platform directory.')
parser.add_argument('platform', help='Platform type')
args = parser.parse_args()

platform_dir = cfg['platforms']['types'][args.platform]['dir']

if not os.path.isabs(platform_dir):
    platform_dir = os.path.join(script_dir, '..', platform_dir)

print(os.path.normpath(platform_dir))

