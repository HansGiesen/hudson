#!/usr/bin/env python3

import argparse
import array
import math
import numpy as np
import os
import zipfile

FIRST_BIN_CONV_LAYER  = 1
FIRST_BIN_DENSE_LAYER = 6
LAST_LAYER            = 8

WIN_HEIGHT = 3
WIN_WIDTH  = 3
WIN_SIZE   = WIN_HEIGHT * WIN_WIDTH

WORD_WIDTH = 64
COEF_WIDTH = 16
WEIGHTS_PER_WORD = WORD_WIDTH // WIN_SIZE
COEFS_PER_WORD = WORD_WIDTH // COEF_WIDTH

def flatten(data, indent = 0):
    space = indent * " "
    if not isinstance(data, list):
        return "{}{}".format(space, data)
    else:
        strings = (flatten(elem, indent + 2) for elem in data)
        text = ",\n".join(strings)
        return "{0}{{\n{1}\n{0}}}".format(space, text)

def load_array(zip_file, idx):
    result = array.array('f')
    with zipfile.ZipFile(zip_file) as zip_file:
        with zip_file.open('arr_{}'.format(idx)) as wt_file:
            data = wt_file.read()
    result.frombytes(data)
    return result

def load_arrays(zip_file, layer):
    wts = load_array(zip_file, 3 * layer)
    ks = load_array(zip_file, 3 * layer + 1)
    hs = load_array(zip_file, 3 * layer + 2)
    return wts, ks, hs

def write_coefs(output_dir, module, wts, ncs):
    with open(os.path.join(output_dir, '{}_wt.h'.format(module)), 'wt') as wt_file:
        wt_file.write("{};".format(flatten(wts.tolist())))
    with open(os.path.join(output_dir, '{}_nc.h'.format(module)), 'wt') as nc_file:
        nc_file.write("{};".format(flatten(ncs.tolist())))

def generate_fp_conv_coefs(args, layer, inputs, outputs):

    conv = args.convolvers[layer]

    raw_wts, raw_ks, raw_hs = load_arrays(args.zip_file, layer)

    idx = 0
    wts = np.zeros((conv, math.ceil(outputs / conv)), np.uint64)
    for out_fmap in range(outputs):
        word = 0
        for in_fmap in range(inputs):
            for win_row in range(WIN_HEIGHT):
                for win_col in range(WIN_WIDTH):
                    value = raw_wts[idx]
                    bit = 0 if value >= 0 else 1
                    word |= bit << (idx % (inputs * WIN_SIZE))
                    idx += 1
        wts[out_fmap % conv][out_fmap // conv] = word

    ncs = np.zeros(outputs // COEFS_PER_WORD, np.int64)
    for out_fmap in range(outputs):
        k = raw_ks[out_fmap]
        h = raw_hs[out_fmap]
        if k != 0.0:
            nc = -h / k
        elif h > 0.0:
            nc = -32767
        else:
            nc = 32767
        nc = np.uint16(math.floor(nc * 2 ** 12))
        ncs[out_fmap // COEFS_PER_WORD] |= nc << (COEF_WIDTH * (out_fmap % COEFS_PER_WORD))

    write_coefs(args.output_dir, 'fp_conv', wts, ncs)

def generate_bin_conv_coefs(args, layer, inputs, outputs):

    conv = args.convolvers[layer]

    raw_wts, raw_ks, raw_hs = load_arrays(args.zip_file, layer)

    idx = 0
    wrd_cnt = math.ceil(outputs * inputs / WEIGHTS_PER_WORD)
    words_per_conv = math.ceil(wrd_cnt / conv)
    wts = np.zeros((conv, words_per_conv), np.uint64)
    for out_fmap in range(outputs):
        for in_fmap in range(inputs):
            fmap_idx = out_fmap * inputs + in_fmap
            conv_idx = fmap_idx % conv
            grp_idx = fmap_idx // conv
            word_idx = grp_idx // WEIGHTS_PER_WORD
            offs = grp_idx % WEIGHTS_PER_WORD
            for win_row in range(WIN_HEIGHT):
                for win_col in range(WIN_WIDTH):
                    value = raw_wts[idx]
                    bit = 0 if value >= 0 else 1
                    shifted_bit = bit << (offs * WIN_SIZE + idx % WIN_SIZE)
                    wts[conv_idx, word_idx] |= np.uint64(shifted_bit)
                    idx += 1

    ncs = np.zeros(outputs // COEFS_PER_WORD, np.int64)
    for out_fmap in range(outputs):
        k = raw_ks[out_fmap]
        h = raw_hs[out_fmap]
        if k != 0.0:
            nc = -h / k
        elif h > 0.0:
            nc = -32767
        else:
            nc = 32767
        nc = np.uint16(math.floor(nc) if nc < 0 else math.ceil(nc))
        ncs[out_fmap // COEFS_PER_WORD] |= nc << (COEF_WIDTH * (out_fmap % COEFS_PER_WORD))

    write_coefs(args.output_dir, 'bin_conv_{}'.format(layer - FIRST_BIN_CONV_LAYER), wts, ncs)

def generate_bin_dense_coefs(args, layer, inputs, outputs):

    conv = args.convolvers[layer]

    raw_wts, raw_ks, raw_hs = load_arrays(args.zip_file, layer)
    
    idx = 0
    wrd_cnt = outputs * inputs // WORD_WIDTH
    wts = np.zeros((conv, wrd_cnt // conv), np.int64)
    for in_fmap in range(inputs):
        for out_fmap in range(outputs):
            value = raw_wts[idx]
            bit = 0 if value >= 0 else 1
            pos = (out_fmap * inputs + in_fmap) // WORD_WIDTH
            wts[pos % conv, pos // conv] |= np.int64(np.uint64(bit << (in_fmap % WORD_WIDTH)))
            idx += 1

    if layer != LAST_LAYER:
        ncs = np.zeros(outputs // COEFS_PER_WORD, np.int64)
        for out_fmap in range(outputs):
            k = raw_ks[out_fmap]
            h = raw_hs[out_fmap]
            if k != 0.0:
                nc = -h / k
            elif h > 0.0:
                nc = -32767
            else:
                nc = 32767
            nc = np.uint16(math.floor(nc) if nc < 0 else math.ceil(nc))
            ncs[out_fmap // COEFS_PER_WORD] |= nc << (COEF_WIDTH * (out_fmap % COEFS_PER_WORD))
    else:
        idx = 0
        ncs = np.zeros(outputs * 2 // COEFS_PER_WORD, np.int64)
        for k, h in zip(raw_ks, raw_hs):
            k = np.uint16(math.floor(k * 2 ** 14))
            ncs[idx // COEFS_PER_WORD] |= k << (COEF_WIDTH * (idx % COEFS_PER_WORD))
            idx += 1
            h = np.uint16(math.floor(h * 2 ** 12))
            ncs[idx // COEFS_PER_WORD] |= h << (COEF_WIDTH * (idx % COEFS_PER_WORD))
            idx += 1

    write_coefs(args.output_dir, 'bin_dense_{}'.format(layer - FIRST_BIN_DENSE_LAYER), wts, ncs)


