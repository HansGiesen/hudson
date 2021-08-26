#!/usr/bin/env python3

import os
import yaml

from argparse import ArgumentParser
from coefs import generate_fp_conv_coefs, generate_bin_conv_coefs, \
                  generate_bin_dense_coefs
from math import ceil, floor
from os.path import join
from sys import exit


word_size = 64
dmem_words = 2048
khmem_words = 256
kh_per_word = 4
kernel_size = 9
pixels_per_phase = 2048
input_cnts = [3, 128, 128, 256, 256, 512, 8192, 1024, 1024]
input_widths = [32, 32, 16, 16, 8, 8, 1, 1, 1]
output_cnts = [128, 128, 256, 256, 512, 512, 1024, 1024, 10]

def str2bool(text):
    if text.lower() == "true":
        return True
    elif text.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_templates(filename):
    templates = {}
    with open(join(args.template_dir, filename), 'r') as template_file:
        for line in template_file:
            if line.startswith('@@@'):
                name = line.split()[1]
                templates[name] = ''
            else:
                templates[name] += line
    return templates

def write_accel(args, conv_map, layer_map):
    templates = load_templates('Accel.cpp')
    with open(join(args.output_dir, 'Accel.cpp'), 'w') as output_file:
        output_file.write(templates['accel_prolog'])
        for module in range(args.bin_conv_cnt):
            write_bin_conv(args, templates, output_file, module, layer_map)
        write_fp_conv(args, templates, output_file)
        for module in range(args.bin_dense_cnt):
            write_bin_dense(args, templates, output_file, module, layer_map)
        write_top(args, templates, output_file, conv_map, layer_map)

def write_bin_conv(args, templates, output_file, module, layer_map):
    if not args.pipeline:
        if args.weights_in_bram:
            output_file.write(templates['bin_conv_hdr_bram'])
        else:
            output_file.write(templates['bin_conv_hdr_dram'])
    else:
        template = templates['bin_conv_hdr_pipe']
        output_file.write(template.format(module))
    output_file.write(templates['bin_conv_0'])
    if args.pipeline:
        output_file.write(templates['bin_conv_hdr_fb'])
    output_file.write(templates['bin_conv_1'])
    if args.weights_in_bram:
        weights_per_word = floor(word_size / kernel_size)
        template = templates['bin_conv_wt_mem']
        for i in layer_map[module + 1]:
            conv = args.convolvers[i]
            cnt = ceil(input_cnts[i] * output_cnts[i] / weights_per_word)
            size = ceil(cnt / conv)
            output_file.write(template.format(i - 1, conv, size, i - 1))
            if conv > 1:
                output_file.write(templates["bin_conv_pragma"].format(i - 1))
    output_file.write(templates['bin_conv_2'])
    if args.pipeline:
        output_file.write(templates['bin_conv_conv_pipe'])
    else:
        output_file.write(templates['bin_conv_conv_seq'])
    trip_cnts_inner = []
    trip_cnts_total = []
    for i in layer_map[module + 1]:
        trip_cnt_outer = input_cnts[i] / args.convolvers[i] 
        trip_cnt_inner = input_widths[i] ** 2 / word_size
        trip_cnt_total = trip_cnt_outer * trip_cnt_inner
        trip_cnts_total.append(trip_cnt_total)
        trip_cnts_inner.append(trip_cnt_inner)
    layer_cnt = len(layer_map[module + 1])
    trip_cnt_inner = int(sum(trip_cnts_inner) / layer_cnt)
    trip_cnt_total = sum(trip_cnts_total) / layer_cnt
    trip_cnt_outer = int(trip_cnt_total / trip_cnt_inner)
    template = templates['bin_conv_3']
    output_file.write(template.format(trip_cnt_outer, trip_cnt_inner))
    if args.weights_in_bram:
        for i in layer_map[module + 1]:
            template = templates['bin_conv_wt_read_bram']
            output_file.write(template.format(i - 1, i - 1))
    else:
        output_file.write(templates['bin_conv_wt_read_dram'])
    output_file.write(templates['bin_conv_4'])
    if args.pipeline:
        output_file.write(templates['bin_conv_dmem_read_pipe'])
    else:
        output_file.write(templates['bin_conv_dmem_read_seq'])
    output_file.write(templates['bin_conv_5'])
    if args.pipeline:
        output_file.write(templates['bin_conv_dmem_write_pipe'])
    else:
        output_file.write(templates['bin_conv_dmem_write_seq'])
    output_file.write(templates['bin_conv_6'])

def write_fp_conv(args, templates, output_file):
    if args.weights_in_bram:
        output_file.write(templates['fp_conv_hdr_bram'])
    else:
        output_file.write(templates['fp_conv_hdr_dram'])
    output_file.write(templates['fp_conv_0'])
    if args.weights_in_bram:
        output_file.write(templates['fp_conv_arrays'])
    output_file.write(templates['fp_conv_1'])
    if not args.pipeline:
      dmem_limit = dmem_words * word_size // input_widths[1] ** 2
      batch_size = min(output_cnts[0], dmem_limit)
      if not args.weights_in_bram:
        wtmem_words = ceil(args.wtmem_size / (word_size // 8))
        wtmem_words_per_bank = ceil(wtmem_words / args.convolvers[0])
        wtmem_words = wtmem_words_per_bank * args.convolvers[0]
        weights_per_word = word_size // kernel_size
        wtmem_limit = wtmem_words * weights_per_word // input_cnts[0]
        kh_mem_limit = khmem_words * kh_per_word
        batch_size = min(batch_size, wtmem_limit, kh_mem_limit)
      while output_cnts[0] % batch_size != 0:
        batch_size -= 1
      output_file.write(templates['fp_conv_2'].format(batch_size))
    output_file.write(templates['fp_conv_3'])

def write_bin_dense(args, templates, output_file, module, layer_map):
    offs = args.bin_conv_cnt + 1
    if not args.pipeline:
        if args.weights_in_bram:
            output_file.write(templates['bin_dense_hdr_bram'])
        else:
            output_file.write(templates['bin_dense_hdr_dram'])
    else:
        template = templates['bin_dense_hdr_pipe']
        output_file.write(template.format(module))
    output_file.write(templates['bin_dense_0'])
    if args.pipeline:
        output_file.write(templates['bin_dense_hdr_fb'])
    output_file.write(templates['bin_dense_1'])
    if args.weights_in_bram:
        template = templates['bin_dense_wt_mem']
        for i in layer_map[module + offs]:
            conv = args.convolvers[i]
            cnt = input_cnts[i] * output_cnts[i] / word_size
            size = ceil(cnt / conv)
            output_file.write(template.format(i - 6, conv, size, i - 6))
            if conv > 1:
                output_file.write(templates["bin_dense_pragma"].format(i - 6))
        output_file.write(templates["bin_dense_kh_mem"])
    output_file.write(templates['bin_dense_2'])
    if not args.pipeline:
        output_file.write(templates['bin_dense_dmem_o_read'])
    trip_cnts = []
    for i in layer_map[module + offs]:
        trip_cnts.append(input_cnts[i] / (args.convolvers[i] * word_size))
    trip_cnt = int(sum(trip_cnts) / len(layer_map[module + offs]))
    output_file.write(templates['bin_dense_3'].format(trip_cnt))
    if args.pipeline:
        output_file.write(templates['bin_dense_dmem_i_read_pipe'])
    else:
        output_file.write(templates['bin_dense_dmem_i_read_seq'])
    output_file.write(templates['bin_dense_4'])
    if args.weights_in_bram:
        for i in layer_map[module + offs]:
            template = templates['bin_dense_wt_read_bram']
            output_file.write(template.format(i - 6, i - 6))
    else:
        output_file.write(templates['bin_dense_wt_read_dram'])
    output_file.write(templates['bin_dense_5'])
    if args.weights_in_bram:
        output_file.write(templates['bin_dense_kh_read_bram'])
    else:
        output_file.write(templates['bin_dense_kh_read_dram'])
    output_file.write(templates['bin_dense_6'])
    if args.pipeline:
        output_file.write(templates['bin_dense_dmem_write_pipe'])
    else:
        output_file.write(templates['bin_dense_dmem_write_seq'])
    output_file.write(templates['bin_dense_7'])

def write_multi_bin_conv(args, templates, output_file, conv_map, layer_map):
    for i in range(1, args.bin_conv_cnt + 1):
        if len(layer_map[i]) == 1:
            continue
        template = templates['multi_bin_conv_hdr']
        output_file.write(template.format(i - 1))
        output_file.write(templates['multi_bin_conv_0'])
        template = templates['multi_bin_conv_arrays']
        for layer in layer_map[i]:
            size = output_cnts[layer] // 4
            output_file.write(template.format(layer - 1, size, layer - 1))
        template = templates['multi_bin_conv_dmem']
        sizes = [1]
        for layer in layer_map[i][1:]:
            size = input_cnts[layer] * input_widths[layer] ** 2
            size = size // word_size if layer > 0 else size
            sizes.append(size // conv_map[i])
        max_size = max(sizes)
        output_file.write(template.format(max_size))
        first = min(layer_map[i]) - 1
        last = max(layer_map[i]) - 1
        template = templates['multi_bin_conv_layers']
        output_file.write(template.format(first, last))
        output_file.write(templates['multi_bin_conv_1'])
        for layer in layer_map[i]:
            template = templates['multi_bin_conv_kh_read']
            output_file.write(template.format(layer - 1, layer - 1))
        output_file.write(templates['multi_bin_conv_call'].format(i - 1))
        output_file.write(templates['multi_bin_conv_2'])

def write_multi_bin_dense(args, templates, output_file, conv_map, layer_map):
    module_cnt = 1 + args.bin_conv_cnt + args.bin_dense_cnt
    for i in range(args.bin_conv_cnt + 1, module_cnt):
        if len(layer_map[i]) == 1:
            continue
        template = templates['multi_bin_dense_hdr']
        output_file.write(template.format(i - args.bin_conv_cnt - 1))
        output_file.write(templates['multi_bin_dense_0'])
        template = templates['multi_bin_dense_dmem']
        sizes = [1]
        for layer in layer_map[i][1:]:
            size = input_cnts[layer] * input_widths[layer] ** 2
            size = size // word_size if layer > 0 else size
            sizes.append(size // conv_map[i])
        max_size = max(sizes)
        output_file.write(template.format(max_size))
        first = min(layer_map[i]) - 6
        last = max(layer_map[i]) - 6
        template = templates['multi_bin_dense_layers']
        output_file.write(template.format(first, last))
        template = templates['multi_bin_dense_1']
        output_file.write(template.format(last - first + 1))
        offs = 1 + args.bin_conv_cnt
        output_file.write(templates['multi_bin_dense_call'].format(i - offs))
        output_file.write(templates['multi_bin_dense_2'])

def write_top(args, templates, output_file, conv_map, layer_map):
    if not args.pipeline:
        output_file.write(templates['top_seq_0'])
        if not args.weights_in_bram:
            output_file.write(templates['top_hdr_wts'])
        output_file.write(templates['top_seq_1'])
        if args.weights_in_bram:
            output_file.write(templates['top_wt_arrays_bram'])
        else:
            output_file.write(templates['top_wt_arrays_dram'])
        output_file.write(templates['top_seq_2'])
        if not args.weights_in_bram:
            output_file.write(templates['top_wt_load'])
        output_file.write(templates['top_seq_3'])
        if not args.weights_in_bram:
            output_file.write(templates['pass_fp_conv_wt'])
        output_file.write(templates['top_seq_4'])
        if args.weights_in_bram:
            output_file.write(templates['top_kh_load_bram'])
        else:
            output_file.write(templates['top_kh_load_dram'])
        output_file.write(templates['top_seq_5'])
        if not args.weights_in_bram:
            output_file.write(templates['pass_bin_conv_wt_1'])
        output_file.write(templates['top_seq_6'])
        if not args.weights_in_bram:
            output_file.write(templates['pass_bin_conv_wt_2'])
        output_file.write(templates['top_seq_7'])
        if not args.weights_in_bram:
            output_file.write(templates['pass_bin_dense_wt'])
        output_file.write(templates['top_seq_8'])
    else:
        write_multi_bin_conv(args, templates, output_file, conv_map, layer_map)
        write_multi_bin_dense(args, templates, output_file, conv_map,
                              layer_map)
        output_file.write(templates['top_pipe_0'])
        template = templates['top_pipe_dmem']
        for i in range(module_cnt):
            layer = layer_map[i][0]
            size = input_cnts[layer] * input_widths[layer] ** 2
            size = size // word_size if layer > 0 else size // 3
            size = size // conv_map[i]
            line = template.format(i, conv_map[i], size, i)
            output_file.write(line)
        line = template.format(module_cnt, 1, 1, module_cnt)
        output_file.write(line)
        output_file.write(templates['top_pipe_1'])
        for i in range(args.bin_conv_cnt):
            if len(layer_map[i + 1]) > 1:
                template = templates['top_pipe_multi_bin_conv']
                output_file.write(template.format(i, i + 1, i + 2))
            else:
                template = templates['top_pipe_bin_conv']
                layer = layer_map[i + 1][0]
                size = output_cnts[layer] // 4
                conv = conv_map[i + 1]
                inputs = input_cnts[layer]
                outputs = output_cnts[layer]
                width_mode = 2 - layer // 2
                norm_mode = 1 + layer % 2
                text = template.format(i + 1, layer,
                                       layer - 1, size, layer - 1,
                                       layer, outputs, layer - 1,
                                       i, i + 1, i + 2, i + 1, i + 1,
                                       inputs, width_mode, norm_mode,
                                       layer - 1)
                output_file.write(text)
        output_file.write(templates['top_pipe_2'])
        offs = args.bin_conv_cnt + 1
        for i in range(args.bin_dense_cnt):
            if len(layer_map[offs + i]) > 1:
                template = templates['top_pipe_multi_bin_dense']
                output_file.write(template.format(i, offs + i, offs + i + 1))
            else:
                template = templates['top_pipe_bin_dense']
                layer = layer_map[i + offs][0]
                inputs = input_cnts[layer]
                outputs = output_cnts[layer]
                last = "LAYER_LAST" if layer == 8 else "LAYER_DENSE"
                text = template.format(offs + i, layer,
                                       i, offs + i, offs + i + 1, offs + i,
                                       offs + i, last, inputs, outputs,
                                       layer - 6)
                output_file.write(text)
        template = templates['top_pipe_3']
        output_file.write(template.format(module_cnt, module_cnt))
        output_file.write(templates['top_pipe_4'])

def write_accel_hdr(args):
    templates = load_templates('Accel.h')
    with open(join(args.output_dir, 'Accel.h'), 'w') as output_file:
        output_file.write(templates['accel_hdr_0'])
        if args.pipeline:
            output_file.write(templates['accel_hdr_prag_pipe'])
            output_file.write(templates['accel_hdr_top_pipe']) 
        else:
            output_file.write(templates['accel_hdr_prag_seq'])
            if not args.weights_in_bram:
                output_file.write(templates['accel_hdr_prag_wt'])
            output_file.write(templates['accel_hdr_top_seq'])
            if not args.weights_in_bram:
                output_file.write(templates['accel_hdr_top_wt'])
            output_file.write(templates['accel_hdr_top_rest']) 
        output_file.write(templates['accel_hdr_1']) 

def write_accel_sched(args):
    templates = load_templates('AccelSchedule.cpp')
    with open(join(args.output_dir, 'AccelSchedule.cpp'), 'w') as output_file:
        output_file.write(templates['accel_sched_0'])
        if not args.weights_in_bram:
            output_file.write(templates['accel_sched_init_wt'])
        output_file.write(templates['accel_sched_1']) 
        if not args.pipeline:
            if not args.weights_in_bram:
                output_file.write(templates['accel_sched_alloc'])
            output_file.write(templates['accel_sched_2'])
            if not args.weights_in_bram:
                output_file.write(templates['accel_sched_copy'])
            output_file.write(templates['accel_sched_3']) 
            if not args.weights_in_bram:
                output_file.write(templates['accel_sched_wts'])
            output_file.write(templates['accel_sched_4']) 
            if not args.weights_in_bram:
                output_file.write(templates['accel_sched_free'])
        output_file.write(templates['accel_sched_5']) 
        if not args.weights_in_bram:
            output_file.write(templates['accel_sched_conv_batch'])
        output_file.write(templates['accel_sched_6']) 
        if not args.weights_in_bram:
            output_file.write(templates['accel_sched_dense_batch'])
        output_file.write(templates['accel_sched_7']) 

def write_bnn(args):
    templates = load_templates('accel_test_bnn.cpp')
    with open(join(args.output_dir, 'accel_test_bnn.cpp'), 'w') as output_file:
        output_file.write(templates['bnn_0'])
        if args.pipeline:
            output_file.write(templates['bnn_dmem_alloc'])
        else:
            output_file.write(templates['bnn_comp_sched'])
        output_file.write(templates['bnn_1'])
        if args.pipeline:
            output_file.write(templates['bnn_pipe'])
        else:
            output_file.write(templates['bnn_seq'])
        output_file.write(templates['bnn_2'])
        if args.pipeline:
            output_file.write(templates['bnn_free_pipe'])
        else:
            output_file.write(templates['bnn_free_seq'])
        output_file.write(templates['bnn_3'])

def check_args(args):
    for layer in range(5):
        phase_cnt = input_cnts[layer + 1] / args.convolvers[layer + 1]
        images_per_phase = pixels_per_phase / input_widths[layer + 1] ** 2
        if phase_cnt % images_per_phase != 0:
            print("Invalid parameter combination.")
            exit(1)

parser = ArgumentParser()
parser.add_argument('project_dir', help='Project dir')
parser.add_argument('cfg_file', help='Configuration file')
args = parser.parse_args()

with open(args.cfg_file, 'r') as input_file:
  data = input_file.read()
cfg = yaml.safe_load(data)['cfg']

args.convolvers = [cfg['CONVOLVERS_%d' % i] for i in range(9)]
args.weights_in_bram = cfg['WEIGHTS_IN_BRAM']
args.pipeline = cfg['PIPELINE']
args.bin_conv_cnt = cfg['BIN_CONV_CNT']
args.bin_dense_cnt = cfg['BIN_DENSE_CNT']
args.wtmem_size = cfg['WT_MEM_SIZE']

args.zip_file = join(args.project_dir, 'params/cifar10_parameters_nb.zip')
args.output_dir = os.path.dirname(os.path.realpath(__file__))
args.template_dir = join(args.output_dir, 'templates')

if args.pipeline:
    args.weights_in_bram = True

if not args.pipeline:
    args.convolvers = [args.convolvers[0]] * 9
    args.bin_conv_cnt = 1
    args.bin_dense_cnt = 1
    conv_map = []
    layer_map = [[0], [1, 2, 3, 4, 5], [6, 7, 8]]
else:
    module_cnt = 1 + args.bin_conv_cnt + args.bin_dense_cnt
    conv_map = args.convolvers[: module_cnt]
    layer_map = [[] for i in range(module_cnt)]
    layer_map[0].append(0)
    for i in range(5):
        module = floor(args.bin_conv_cnt * i / 5)
        layer_map[module + 1].append(i + 1)
        args.convolvers[i + 1] = conv_map[module + 1]
    offs = 1 + args.bin_conv_cnt
    for i in range(3):
        module = floor(args.bin_dense_cnt * i / 3)
        layer_map[module + offs].append(i + 6)
        args.convolvers[i + 6] = conv_map[offs + module]

check_args(args)

generate_fp_conv_coefs(args, 0, 3, 128)
generate_bin_conv_coefs(args, 1, 128, 128)
generate_bin_conv_coefs(args, 2, 128, 256)
generate_bin_conv_coefs(args, 3, 256, 256)
generate_bin_conv_coefs(args, 4, 256, 512)
generate_bin_conv_coefs(args, 5, 512, 512)
generate_bin_dense_coefs(args, 6, 8192, 1024)
generate_bin_dense_coefs(args, 7, 1024, 1024)
generate_bin_dense_coefs(args, 8, 1024, 10)

write_accel(args, conv_map, layer_map)
write_accel_hdr(args)
write_accel_sched(args)
write_bnn(args)
