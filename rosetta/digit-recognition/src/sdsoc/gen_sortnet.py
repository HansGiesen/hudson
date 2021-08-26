#!/usr/bin/env python3
# -*- coding: utf-8 -*-def gen():

# This code is based on work by Yuichi Sugiyama.
#   https://github.com/mmxsrup/bitonic-sort

import argparse
import math
import os
import yaml

def gen(filename, input_cnt, swap_cnt):

    with open(filename, 'w') as f:

        main_s = []
        f.write('#include "swap.hpp"\n\n')

        block = 2
        while (block <= input_cnt):
            step = int(block / 2)
            while (step >= 1):

                s = 'void block%d_step%d_net(SortData data[SORT_INPUT_CNT],' % (block, step)
                s += ' SortId ids[SORT_INPUT_CNT]) {\n'
                s += '  #pragma HLS inline\n'
                f.write(s)

                main_s.append('      block%d_step%d_net(data, ids);\n' % (block, step))

                for idx1 in range(0, input_cnt):
                    idx2 = idx1 ^ step
                    if (idx1 >= idx2):
                        continue
                    direction = 'true' if (idx1 & block) != 0 else 'false'
                    s = '  swap(%s, data[%d], data[%d], ids[%d], ids[%d]);\n' % \
                        (direction, idx1, idx2, idx1, idx2)
                    f.write(s)

                s = '}\n\n'
                f.write(s)

                step = int(step / 2)
            block = block * 2

        f.write('void sorting_network(SortData data[SORT_INPUT_CNT], ')
        f.write('SortId ids[SORT_INPUT_CNT]) {\n')
        f.write('  #pragma HLS allocation instances=swap limit=%i\n' % swap_cnt)
        f.write('  for (int i = 0; i < %i; i++) {\n' % len(main_s))
        for i, s in enumerate(main_s):
            if i == 0:
                f.write('    if (i == %i) {\n' % i)
            elif i < len(main_s) - 1:
                f.write('    else if (i == %i) {\n' % i)
            else:
                f.write('    else {\n')
            f.write(s)
            f.write('    }\n')
        f.write('  }\n')
        f.write('}\n')


def gen_header(filename_header, input_cnt):
    with open(filename_header, 'w') as f:
        f.write('#ifndef __SORTING_NETWORK_HPP__\n')
        f.write('#define __SORTING_NETWORK_HPP__\n\n')
        f.write('#include <ap_int.h>\n\n')
        f.write('#define SORT_INPUT_CNT (%d)\n\n' % input_cnt)
        f.write('typedef ap_uint<9> SortData;\n')
        f.write('typedef ap_uint<4> SortId;\n\n')
        f.write('void sorting_network(SortData data[SORT_INPUT_CNT],')
        f.write(' SortId ids[SORT_INPUT_CNT]);\n')
        f.write('\n#endif\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate bitonic sort network.')
    parser.add_argument('project_dir', help='Project dir')
    parser.add_argument('cfg_file', help='Configuration file')
    args = parser.parse_args()
    
    with open(args.cfg_file, 'r') as input_file:
        data = input_file.read()
    cfg = yaml.safe_load(data)['cfg']

    input_cnt = 2 ** math.ceil(math.log2(30 * cfg['PAR_FACTOR']))
    swap_cnt = cfg['SWAP_CNT']

    output_dir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(output_dir, 'sorting_network.cpp')
    filename_header = os.path.join(output_dir, 'sorting_network.hpp')

    gen(filename, input_cnt, swap_cnt)
    gen_header(filename_header, input_cnt)
