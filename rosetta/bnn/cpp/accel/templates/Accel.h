@@@ accel_hdr_0
#ifndef ACCEL_ACCEL_H
#define ACCEL_ACCEL_H

#include <cstddef>
#include <stdlib.h>   // include this before sds_lib.h for size_t

#include "Typedefs.h"
#include "Debug.h"
#include "Common.h"

#ifdef __SDSCC__
  #include "sds_lib.h"
  #define MEM_ALLOC(size) sds_alloc(size)
  #define MEM_FREE(ptr) sds_free(ptr)
#else
  #define MEM_ALLOC(size) malloc(size)
  #define MEM_FREE(ptr) free(ptr)
#endif

//-------------------------------------------------------------------
// Constants
//-------------------------------------------------------------------

const unsigned CONVOLVERS = CONVOLVERS_0;

const unsigned WORD_SIZE = 64;
const unsigned WT_SIZE = 9;
const unsigned CONV_W_PER_WORD = 7;
const unsigned CONV1_W_PER_WORD = 3;
const unsigned KH_PER_WORD = 4;
const unsigned K = 3;
const unsigned BYTES_PER_WORD = WORD_SIZE / 8;
const unsigned C_WT_WORDS   = ((WT_MEM_SIZE+BYTES_PER_WORD-1)/BYTES_PER_WORD + CONVOLVERS-1) / CONVOLVERS;  // wt words per convolver
const unsigned WT_WORDS     = C_WT_WORDS*CONVOLVERS;
const unsigned KH_WORDS     = 1024 / KH_PER_WORD; // HG: Enough space for all coefficients of the largest layer

const unsigned PIX_PER_PHASE = 2*32*32;

const unsigned MAX_WIDTH = WORD_SIZE;
const unsigned BANK_WIDTH = 8;
const unsigned LOG_BANK_WIDTH = 3;

const unsigned CONV_ROWS = 3;
const unsigned CONV_COLS = BANK_WIDTH+2;
const unsigned CONV_BANKS = WORD_SIZE / BANK_WIDTH;

const unsigned DMEM_WORDS = 128 * 32 * 32 / WORD_SIZE;
const unsigned C_DMEM_WORDS = DMEM_WORDS / CONVOLVERS;

const unsigned DMEM_WORDS_0 = 32 * 32;
const unsigned DMEM_WORDS_1 = 128 * 32 * 32 / WORD_SIZE;
const unsigned DMEM_WORDS_2 = 128 * 16 * 16 / WORD_SIZE;
const unsigned DMEM_WORDS_3 = 256 * 16 * 16 / WORD_SIZE;
const unsigned DMEM_WORDS_4 = 256 * 8 * 8 / WORD_SIZE;
const unsigned DMEM_WORDS_5 = 512 * 8 * 8 / WORD_SIZE;
const unsigned DMEM_WORDS_6 = 8192 / WORD_SIZE;
const unsigned DMEM_WORDS_7 = 1024 / WORD_SIZE;
const unsigned DMEM_WORDS_8 = 1024 / WORD_SIZE;
const unsigned DMEM_WORDS_9 = 1;

const unsigned C_DMEM_WORDS_0 = DMEM_WORDS_0 / CONVOLVERS_0;
const unsigned C_DMEM_WORDS_1 = DMEM_WORDS_1 / CONVOLVERS_1;
const unsigned C_DMEM_WORDS_2 = DMEM_WORDS_2 / CONVOLVERS_2;
const unsigned C_DMEM_WORDS_3 = DMEM_WORDS_3 / CONVOLVERS_3;
const unsigned C_DMEM_WORDS_4 = DMEM_WORDS_4 / CONVOLVERS_4;
const unsigned C_DMEM_WORDS_5 = DMEM_WORDS_5 / CONVOLVERS_5;
const unsigned C_DMEM_WORDS_6 = DMEM_WORDS_6 / CONVOLVERS_6;
const unsigned C_DMEM_WORDS_7 = DMEM_WORDS_7 / CONVOLVERS_7;
const unsigned C_DMEM_WORDS_8 = DMEM_WORDS_8 / CONVOLVERS_8;
const unsigned C_DMEM_WORDS_9 = 1;

const unsigned WT_L_0 = 3 * 128;
const unsigned WT_L_1 = 128 * 128;
const unsigned WT_L_2 = 128 * 256;
const unsigned WT_L_3 = 256 * 256;
const unsigned WT_L_4 = 256 * 512;
const unsigned WT_L_5 = 512 * 512;
const unsigned WT_L_6 = 8192 * 1024;
const unsigned WT_L_7 = 1024 * 1024;
const unsigned WT_L_8 = 1024 * 10;

const unsigned C_WT_WORDS_0 = ((WT_L_0 + CONV1_W_PER_WORD - 1) / CONV1_W_PER_WORD + CONVOLVERS_0 - 1) / CONVOLVERS_0;
const unsigned C_WT_WORDS_1 = ((WT_L_1 + CONV_W_PER_WORD - 1) / CONV_W_PER_WORD + CONVOLVERS_1 - 1) / CONVOLVERS_1;
const unsigned C_WT_WORDS_2 = ((WT_L_2 + CONV_W_PER_WORD - 1) / CONV_W_PER_WORD + CONVOLVERS_2 - 1) / CONVOLVERS_2;
const unsigned C_WT_WORDS_3 = ((WT_L_3 + CONV_W_PER_WORD - 1) / CONV_W_PER_WORD + CONVOLVERS_3 - 1) / CONVOLVERS_3;
const unsigned C_WT_WORDS_4 = ((WT_L_4 + CONV_W_PER_WORD - 1) / CONV_W_PER_WORD + CONVOLVERS_4 - 1) / CONVOLVERS_4;
const unsigned C_WT_WORDS_5 = ((WT_L_5 + CONV_W_PER_WORD - 1) / CONV_W_PER_WORD + CONVOLVERS_5 - 1) / CONVOLVERS_5;
const unsigned C_WT_WORDS_6 = (WT_L_6 / WORD_SIZE + CONVOLVERS_6 - 1) / CONVOLVERS_6;
const unsigned C_WT_WORDS_7 = (WT_L_7 / WORD_SIZE + CONVOLVERS_7 - 1) / CONVOLVERS_7;
const unsigned C_WT_WORDS_8 = (WT_L_8 / WORD_SIZE + CONVOLVERS_8 - 1) / CONVOLVERS_8;

const unsigned WT_WORDS_0 = C_WT_WORDS_0 * CONVOLVERS_0;
const unsigned WT_WORDS_1 = C_WT_WORDS_1 * CONVOLVERS_1;
const unsigned WT_WORDS_2 = C_WT_WORDS_2 * CONVOLVERS_2;
const unsigned WT_WORDS_3 = C_WT_WORDS_3 * CONVOLVERS_3;
const unsigned WT_WORDS_4 = C_WT_WORDS_4 * CONVOLVERS_4;
const unsigned WT_WORDS_5 = C_WT_WORDS_5 * CONVOLVERS_5;
const unsigned WT_WORDS_6 = C_WT_WORDS_6 * CONVOLVERS_6;
const unsigned WT_WORDS_7 = C_WT_WORDS_7 * CONVOLVERS_7;
const unsigned WT_WORDS_8 = C_WT_WORDS_8 * CONVOLVERS_8;

const unsigned KH_WORDS_0 = 128 / KH_PER_WORD;
const unsigned KH_WORDS_1 = 128 / KH_PER_WORD;
const unsigned KH_WORDS_2 = 256 / KH_PER_WORD;
const unsigned KH_WORDS_3 = 256 / KH_PER_WORD;
const unsigned KH_WORDS_4 = 512 / KH_PER_WORD;
const unsigned KH_WORDS_5 = 512 / KH_PER_WORD;
const unsigned KH_WORDS_6 = 1024 / KH_PER_WORD;
const unsigned KH_WORDS_7 = 1024 / KH_PER_WORD;
const unsigned KH_WORDS_8 = 10 * 2 / KH_PER_WORD;

//-------------------------------------------------------------------
// Typedefs
//-------------------------------------------------------------------
enum LayerTypeEnum {LAYER_CONV1, LAYER_CONV, LAYER_DENSE, LAYER_LAST};

typedef ap_int<WORD_SIZE> Word;
typedef ap_int<WT_SIZE> WtType;
typedef ap_uint<17> Address;
typedef ap_int<12> ConvSum;
typedef ap_int<5> ConvOut;
typedef ap_uint<10> IdxType;
typedef ap_fixed<16,4> C1Comp;
typedef ap_int<16> NormComp;
typedef ap_int<16> DenseSum;
typedef ap_fixed<16,12> DenseNorm;

typedef ap_fixed<20,2, AP_RND> C1InputType;
typedef ap_fixed<24,6, AP_RND> C1ConvType;


//-------------------------------------------------------------------
// Template functions
//-------------------------------------------------------------------
template<typename T1, typename T2>
void load_kh(T1& comp, const T2 kh_mem[KH_WORDS], Address idx) {
  Word kh_word = kh_mem[idx/KH_PER_WORD];
  IdxType off = idx % KH_PER_WORD;
  if (off == 0)
    comp(15,0) = kh_word(15, 0);
  else if (off == 1)
    comp(15,0) = kh_word(31,16);
  else if (off == 2)
    comp(15,0) = kh_word(47,32);
  else
    comp(15,0) = kh_word(63,48);
}

//-------------------------------------------------------------------
// Accelerator synthesizable top-level functions
//-------------------------------------------------------------------

@@@ accel_hdr_prag_seq
#pragma SDS data copy(dmem_i[0:input_words], dmem_o[0:output_words])
#pragma SDS data access_pattern(dmem_i:SEQUENTIAL, dmem_o:SEQUENTIAL)
#pragma SDS data mem_attribute(dmem_i:PHYSICAL_CONTIGUOUS, dmem_o:PHYSICAL_CONTIGUOUS)
@@@ accel_hdr_prag_wt
#pragma SDS data copy(wt_i[0:wt_words], kh_i[0:kh_words])
#pragma SDS data access_pattern(wt_i:SEQUENTIAL, kh_i:SEQUENTIAL)
#pragma SDS data mem_attribute(wt_i:PHYSICAL_CONTIGUOUS, kh_i:PHYSICAL_CONTIGUOUS)
@@@ accel_hdr_prag_pipe
#pragma SDS data access_pattern(dmem_i:SEQUENTIAL)
#pragma SDS data mem_attribute(dmem_i:PHYSICAL_CONTIGUOUS)
#pragma SDS data buffer_depth(dmem_i:9, pred:9)
@@@ accel_hdr_top_seq
void top(
    int layer,
@@@ accel_hdr_top_wt
    Word wt_i[WT_WORDS],
    Word kh_i[KH_WORDS],
    const Address wt_words,
    const Address kh_words,
@@@ accel_hdr_top_rest
    Word dmem_i[DMEM_WORDS],
    Word dmem_o[DMEM_WORDS],
    const Address    n_inputs,
    const Address    n_outputs,
    const Address    input_words,
    const Address    output_words,
    const ap_uint<3> layer_mode,  // [0]='new layer', [2:1]='conv1,conv,dense'
    const ap_uint<1> dmem_mode,   // 0 means dmem[0] is input
    const ap_uint<2> width_mode,  // 0=8'b, 1=16'b, 2=32'b
    const ap_uint<2> norm_mode    // 0='do nothing', 1='do norm', 2='do pool'
@@@ accel_hdr_top_pipe
void top(
    Word dmem_i[DMEM_WORDS_0],
    ap_uint<4> & pred
@@@ accel_hdr_1
);

#endif
