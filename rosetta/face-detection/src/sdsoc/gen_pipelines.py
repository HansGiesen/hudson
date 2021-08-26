#!/usr/bin/env python3

height = 240
width = 320
win_size = 25
factor = 1.2

import argparse
import os
import yaml

parser = argparse.ArgumentParser(description='Generate pipeline header file.')
parser.add_argument('project_dir', help='Project dir')
parser.add_argument('cfg_file', help='Configuration file')

args = parser.parse_args()
    
with open(args.cfg_file, 'r') as input_file:
  data = input_file.read()
cfg = yaml.safe_load(data)['cfg']

pipeline_cnt = cfg['PIPELINE_CNT']

sizes = []
while True:
  height /= factor
  width /= factor
  if height <= win_size or width <= win_size:
    break
  sizes.append(round(width) * round(height))

imgs = [[] for i in range(pipeline_cnt)]
pipeline_sizes = [0] * pipeline_cnt
for img, size in enumerate(sizes):
  pipeline = pipeline_sizes.index(min(pipeline_sizes))
  pipeline_sizes[pipeline] += size
  imgs[pipeline].append(img)

lens = list(map(len, imgs))
max_len = max(lens)

indices = ['{{{}}}'.format(', '.join(map(str, elem))) for elem in imgs]
indices = ', '.join(indices)
lens = ', '.join(map(str, lens))

output_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(output_dir, 'pipelines.h'), 'w') as output_file:
  output_file.write('unsigned pipeline_idx[][{}] = {{{}}};\n'.format(max_len,
                                                                     indices))
  output_file.write('unsigned pipeline_len[] = {{{}}};\n'.format(lens))
