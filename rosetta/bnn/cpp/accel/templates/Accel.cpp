@@@ accel_prolog
#include <iostream>
#include <iomanip>
#include "Accel.h"
#include "AccelPrint.h"

const static Word m1("0x5555555555555555", 16);
const static Word m2("0x3333333333333333", 16);
const static Word m4("0x0f0f0f0f0f0f0f0f", 16);
const static Word h01("0x0101010101010101", 16);

// -----------------------------------------------------------------------
// Hardware-specific print helpers
// -----------------------------------------------------------------------
template<typename T>
void print_ap_bits(const T& in, const unsigned W) {
  printf ("   ");
  for (unsigned i = 0; i < W; ++i)
    printf ("%3d", in[i] ? -1 : 0);
  printf ("\n");
}

template<typename T>
void print_params(T params[CONVOLVERS][K][K]) {
  for (unsigned m = 0; m < CONVOLVERS; ++m) {
    for (unsigned wr = 0; wr < K; ++wr) {
      for (unsigned wc = 0; wc < K; ++wc) {
        printf ("%3d", (params[m][wr][wc]==0) ? 0 : 1);
      }
      printf("\n");
    }
    printf("--\n");
  }
}

template<typename T>
void print_line_buffer_m(T lbuf[CONV_BANKS]) {
  for (unsigned wr = 0; wr < CONV_ROWS; ++wr) {
  for (unsigned bank = 0; bank < CONV_BANKS; ++bank) {
    for (unsigned wc = 0; wc < CONV_COLS; ++wc) {
      printf ("%3d", lbuf[bank][wr][wc].to_int());
    }
    printf (" |");
  }
  printf ("\n");
  }
}

// Note the different layer numbering here:
//   Layer 0 = fp_conv input
//   Layer 1 = fp_conv output
//   ...
//   Layer 9 = bin_dense output
template<int convolvers, int dmem_words>
  void print_dmem(Word (& dmem)[convolvers][dmem_words], int layer)
{
  if (BATCH_SIZE > 1)
    return;
  if (layer < N_LAYERS)
  {
    if (layer == 0)
    {
      printf("Layer %i input: 32 x 32 x 3 x 20 bits (1024 words)\n", layer + 1);
      for (int i = 0; i < 32 * 32; i++)
        printf("%i %i: %016lX\n", layer, i, dmem[i % convolvers][i / convolvers].to_ulong());
    }
    else if (layer >= 1 && layer <= 5)
    {
      int inputs = M_tab[layer];
      int width = S_tab[layer];
      int words_per_input = width * width / WORD_SIZE;
      int words = inputs * words_per_input;
      printf("Layer %i input: %i x %i x %i x 1 bits (%i words)\n", layer + 1, inputs, width, width, words);
      for (int i = 0; i < words; i++)
      {
        int input_idx = i / words_per_input;
        int word_idx = i % words_per_input;
        int bank_idx = input_idx % convolvers;
        int bank_offs = input_idx / convolvers;
        printf("%i %i: %016lX\n", layer, i, dmem[bank_idx][bank_offs * words_per_input + word_idx].to_ulong());
      }
    }
    else
    {
      int inputs = M_tab[layer];
      int words = inputs / WORD_SIZE;
      printf("Layer %i input: %i x 1 bits (%i words)\n", layer + 1, inputs, words);
      for (int i = 0; i < words; i++)
        printf("%i %i: %016lX\n", layer, i, dmem[i % convolvers][i / convolvers].to_ulong());
    }
  }
  else
  {
    printf("Layer %i output: 4 bits (1 word)\n", layer);
    printf("%i 0: %016lX\n", layer, dmem[0][0].to_ulong());
  }
}


TwoBit encode_bit(const Bit& b) {
#pragma HLS INLINE
  return (b == 0) ? TwoBit(1) : TwoBit(-1);
}

// -----------------------------------------------------------------------
// Conv
// -----------------------------------------------------------------------
ConvOut conv3x3b(
    const TwoBit line_buffer_m[CONV_BANKS][CONV_ROWS][CONV_COLS],
    const Bit conv_params_m[K][K],
    const ap_uint<4> bank,
    const IdxType cc
) {
#pragma HLS INLINE
  ConvOut sum = 0;
  for (ap_uint<2> kr = 0; kr < K; ++kr) {
    for (ap_uint<2> kc = 0; kc < K; ++kc) {
      TwoBit data = line_buffer_m[bank][kr][cc+kc];
      const Bit& wt = conv_params_m[2-kr][2-kc];
      data[1] = (wt & data[0]) ^ data[1];
      sum += data;
    }
  }
  return sum;
}

// -----------------------------------------------------------------------
// Produce 32 elements of conv results
// -----------------------------------------------------------------------
void conv_word(
    const TwoBit line_buffer_m[CONV_BANKS][CONV_ROWS][CONV_COLS],
    const Bit conv_params_m[K][K],
    ConvOut conv_out_buffer_m[WORD_SIZE]
) {
#pragma HLS PIPELINE
  for (ap_uint<4> bank = 0; bank < CONV_BANKS; ++bank) {
    for (ap_uint<4> cc = 0; cc < BANK_WIDTH; ++cc) {
      conv_out_buffer_m[bank*BANK_WIDTH+cc] = conv3x3b( line_buffer_m, conv_params_m, bank, cc );
    }
  }
}

// -----------------------------------------------------------------------
// Process each line in a word, we need to outline this loop to
// avoid false control dependencies in Vivado HLS
// -----------------------------------------------------------------------
void process_word(
    const TwoBit  word_buffer_m[CONV_BANKS][CONV_COLS],
    const TwoBit  old_word_buffer_m[CONV_BANKS][CONV_COLS],
    const bool lb[CONV_BANKS],
    const bool rb[CONV_BANKS],
    TwoBit  line_buffer_m[CONV_BANKS][CONV_ROWS][CONV_COLS],
    const   Bit conv_params_m[K][K],
    ConvOut conv_out_buffer_m[WORD_SIZE],
    const   ap_uint<3> log_width,
    const   ap_uint<6> words_per_image,
    const   IdxType wrd
) {
#pragma HLS INLINE
  // slices_per_line = width / BANK_WIDTH
  const ap_uint<5> slices_per_line = 1 << (log_width - LOG_BANK_WIDTH);
  const bool first_wrd = (wrd == 0);
  const bool last_wrd = (wrd == words_per_image);
  DB_PRINT(4, "process word %d, spl=%d\n", wrd.to_int(), slices_per_line.to_int());

  // Prologue
  // Update bottom row, slices are shifted left. Some slices copied from previous word (middle row)
  for (ap_uint<4> bank = 0; bank < CONV_BANKS; ++bank) {
    ap_int<6> s_idx = bank + slices_per_line - CONV_BANKS;
    if (s_idx < 0) {
      // set to zero or copy from old word (middle row)
      for (ap_uint<4> cc = 1; cc < CONV_COLS-1; ++cc) {
        line_buffer_m[bank][CONV_ROWS-1][cc] = old_word_buffer_m[CONV_BANKS+s_idx][cc];
      }
      line_buffer_m[bank][CONV_ROWS-1][0          ] = lb[bank] ? TwoBit(0) : old_word_buffer_m[CONV_BANKS+s_idx][0];
      line_buffer_m[bank][CONV_ROWS-1][CONV_COLS-1] = rb[bank] ? TwoBit(0) : old_word_buffer_m[CONV_BANKS+s_idx][CONV_COLS-1];
    } else {
      // fill from new word
      for (ap_uint<4> cc = 1; cc < CONV_COLS-1; ++cc) {
        line_buffer_m[bank][CONV_ROWS-1][cc] = (last_wrd) ? TwoBit(0) : word_buffer_m[s_idx][cc];
      }
      line_buffer_m[bank][CONV_ROWS-1][0          ] = (last_wrd || lb[bank]) ? TwoBit(0) : word_buffer_m[s_idx][0];
      line_buffer_m[bank][CONV_ROWS-1][CONV_COLS-1] = (last_wrd || rb[bank]) ? TwoBit(0) : word_buffer_m[s_idx][CONV_COLS-1];
    }
  }
  
  DB(4,
    printf("Accel lbuf wrd%d before conv:\n", wrd.to_int());
    print_line_buffer_m(line_buffer_m);
  );

  // Convolution
  conv_word( line_buffer_m, conv_params_m, conv_out_buffer_m );
  
  // Update
  // Fill line buffer with lines from the new word
  for (ap_uint<4> bank = 0; bank < CONV_BANKS; ++bank) {
    // --------------------------------------------------------------
    // Top row, slices are shifted right by slices_per_line
    ap_int<6> s_idx0 = bank - slices_per_line;
    if (s_idx0 >= 0) {
      // slice from input word
      for (ap_uint<4> cc = 1; cc < CONV_COLS-1; ++cc) {
        line_buffer_m[bank][0][cc] = word_buffer_m[s_idx0][cc];
      }
      line_buffer_m[bank][0][0          ] = lb[bank] ? TwoBit(0) : word_buffer_m[s_idx0][0];
      line_buffer_m[bank][0][CONV_COLS-1] = rb[bank] ? TwoBit(0) : word_buffer_m[s_idx0][CONV_COLS-1];
    } else {
      // set to zero or copy from old word (middle row)
      for (ap_uint<4> cc = 1; cc < CONV_COLS-1; ++cc) {
        line_buffer_m[bank][0][cc] = (first_wrd) ? TwoBit(0) : old_word_buffer_m[CONV_BANKS+s_idx0][cc];
      }
      line_buffer_m[bank][0][0          ] = (first_wrd || lb[bank]) ? TwoBit(0) : old_word_buffer_m[CONV_BANKS+s_idx0][0];
      line_buffer_m[bank][0][CONV_COLS-1] = (first_wrd || rb[bank]) ? TwoBit(0) : old_word_buffer_m[CONV_BANKS+s_idx0][CONV_COLS-1];
    }

    // --------------------------------------------------------------
    // Middle row, simply copy the word into the line buffer
    for (ap_uint<4> cc = 1; cc < CONV_COLS-1; ++cc) {
      line_buffer_m[bank][1][cc] = word_buffer_m[bank][cc];
    }
    // Fill end buffer bits
    line_buffer_m[bank][1][0          ] = lb[bank] ? TwoBit(0) : word_buffer_m[bank][0];
    line_buffer_m[bank][1][CONV_COLS-1] = rb[bank] ? TwoBit(0) : word_buffer_m[bank][CONV_COLS-1];
  }

  DB(4,
    printf("Accel lbuf wrd%d after conv:\n", wrd.to_int());
    print_line_buffer_m(line_buffer_m);
  );
}

// -----------------------------------------------------------------------
// A single PE reads from all inputs and weights to generate a single
// output feature map.
// * Make sure this function gets inlined by VHLS, or cosim may fail!
// -----------------------------------------------------------------------
@@@ bin_conv_hdr_dram
template <int conv_i, int conv_o, int c_wt_words, int c_dmem_i_words, int c_dmem_o_words> void bin_conv(
    Word (& wt_mem)[conv_i][c_wt_words],
@@@ bin_conv_hdr_bram
template <int conv_i, int conv_o, int c_dmem_i_words, int c_dmem_o_words> void bin_conv(
@@@ bin_conv_hdr_pipe
template <int conv_i, int conv_o, int c_dmem_i_words, int c_dmem_o_words, int c_dmem_fb_words> void bin_conv_{}(
@@@ bin_conv_0
    NormComp nc,
    Word (& dmem_i)[conv_i][c_dmem_i_words],
    Word (& dmem_o)[conv_o][c_dmem_o_words],
@@@ bin_conv_hdr_fb
    Word (& dmem_fb_i)[conv_i][c_dmem_fb_words],
    Word (& dmem_fb_o)[conv_i][c_dmem_fb_words],
    const ap_uint<1> read_fb,
    const ap_uint<1> write_fb,
@@@ bin_conv_1
    const unsigned   n_inputs,
    const Address    o_index,
    const ap_uint<1> new_batch,
    const unsigned   width_mode,  // 0=8'b, 1=16'b, 2=32'b
    const ap_uint<2> norm_mode,   // 0='do nothing', 1='do norm', 2='do pool'
    int layer
) {
//  if (layer == 2)
//    printf("%u %u %u %u %u %u %04X\n", layer, n_inputs, o_index.to_uint(), new_batch.to_uint(), width_mode, norm_mode.to_uint(), nc(15, 0).to_uint());
  const unsigned images_per_conv = n_inputs / conv_i;
  const ap_uint<3> log_width = width_mode + LOG_BANK_WIDTH;
  const unsigned words_per_image = 1 << (2*width_mode);

@@@ bin_conv_wt_mem
  static unsigned long long wt_mem_{}[{}][{}] =
#include "bin_conv_{}_wt.h"
@@@ bin_conv_pragma
#pragma HLS ARRAY_PARTITION variable=wt_mem_{} complete dim=1
@@@ bin_conv_2
  // ---------------------------------------------------------------------
  // buffers
  // ---------------------------------------------------------------------
  TwoBit  line_buffer[conv_i][CONV_BANKS][CONV_ROWS][CONV_COLS];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=0
  Bit     conv_params[conv_i][K][K];
#pragma HLS ARRAY_PARTITION variable=conv_params complete dim=0
  ConvSum fixed_buffer[32][WORD_SIZE];
#pragma HLS ARRAY_PARTITION variable=fixed_buffer complete dim=2
  // per-convolver buffers
  TwoBit  word_buffer[conv_i][CONV_BANKS][CONV_COLS];
#pragma HLS ARRAY_PARTITION variable=word_buffer complete dim=0
  TwoBit  old_word_buffer[conv_i][CONV_BANKS][CONV_COLS];
#pragma HLS ARRAY_PARTITION variable=old_word_buffer complete dim=0
  ConvOut conv_out_buffer[conv_i][WORD_SIZE];
#pragma HLS ARRAY_PARTITION variable=conv_out_buffer complete dim=0
  // edge padding flag bits
  bool lb[CONV_BANKS];
#pragma HLS ARRAY_PARTITION variable=lb complete dim=0
  bool rb[CONV_BANKS];
#pragma HLS ARRAY_PARTITION variable=rb complete dim=0

  static Address wt_addr = 0;           // address of weight word
  static ap_uint<3> wt_offset = 0;      // offset 0..6 of param
  if (new_batch != 0) { wt_addr = 0; wt_offset = 0; }

#ifndef __SYNTHESIS__
  for (int i = 0; i < conv_i; i++)  
    for (int j = 0; j < CONV_BANKS; j++)
      for (int k = 0; k < CONV_ROWS; k++)
        for (int l = 0; l < CONV_COLS; l++)
          line_buffer[i][j][k][l] = 0;
  
  for (int i = 0; i < conv_i; i++)  
    for (int j = 0; j < CONV_BANKS; j++)
      for (int l = 0; l < CONV_COLS; l++)
        old_word_buffer[i][j][l] = 0;
#endif

  // ---------------------------------------------------------------------
  // Calculate edge padding flag bits
  const ap_uint<4> log_slice = log_width - LOG_BANK_WIDTH;
  const ap_uint<4> w_div_8 = (1 << log_width) >> 3;
  assert (w_div_8 > 0);
  ap_uint<4> mask = ~ap_uint<4>(0);   // set mask to all 1s
  mask = mask >> (4-log_slice);
  for (ap_uint<4> bank = 0; bank < CONV_BANKS; ++bank) {
    #pragma HLS unroll
    const ap_uint<4> x = bank & mask;
    lb[bank] = (x == 0);          // (bank % w_div_8) == 0
    rb[bank] = (x+1 == w_div_8);  // (bank % w_div_8) == w_div_8-1
  }

@@@ bin_conv_conv_seq
  const int conv = conv_o;
@@@ bin_conv_conv_pipe
  const int conv = write_fb ? conv_i : conv_o;
@@@ bin_conv_3
  const Address bank_idx = o_index % conv;
  const Address bank_off = o_index / conv;
  const ap_uint<5> pool_width = 1 << (log_width-1);
  static Word outword;
  Word poolword;

  // Load a word each iteration, and then process it
  // We need 1 extra "empty" iteration per image to do the loop epilogue.
  LOOP_IMAGES:
  for (ap_uint<10> img = 0; img < images_per_conv; img++)
  {{
#pragma HLS loop_tripcount max={}
    LOOP_WORDS:
    for (ap_uint<8> wrd = 0; wrd <= words_per_image; wrd++) {{
#pragma HLS loop_tripcount max={}
#pragma HLS PIPELINE
      // First word of an image
      if (wrd == 0) {{
        Word wt_word_buffer[conv_i];
#pragma HLS ARRAY_PARTITION variable=wt_word_buffer complete dim=0

        // -------------------------------------------------------------------
        // Load param word
        // Each word contains CONV_W_PER_WORD weight filters, after we use
        // them all we should load the next word
        // -------------------------------------------------------------------
        LOOP_WT_WORDS:
        for (IdxType m = 0; m < conv_i; ++m) {{
          /*if (wt_offset == 0)
            wt_word_buffer[m] = wt_mem[m][wt_addr];
          else
            wt_word_buffer[m] = wt_word_buffer[m] >> WT_SIZE;
          */
          Word wt_word;
@@@ bin_conv_wt_read_dram
          wt_word = wt_mem[m][wt_addr];
@@@ bin_conv_wt_read_bram
          if (layer == {})
            wt_word = wt_mem_{}[m][wt_addr];
@@@ bin_conv_4
          wt_word_buffer[m] = wt_word >> ap_uint<6>(WT_SIZE*wt_offset);
//          if (layer == 2)
//            printf("WT %i %i %i %i %i: %0lX\n", p.to_int(), count.to_int(), m.to_int(), wt_addr.to_int(), wt_offset.to_int(), wt_word_buffer[m].to_ulong());
        }
        if (wt_offset == CONV_W_PER_WORD-1) {
          ++wt_addr;
          wt_offset = 0;
        } else {
          ++wt_offset;
        }
        //print_wt_word(wt_word_buffer[0]);

        // -------------------------------------------------------------------
        // Load params
        // Each word contains CONV_W_PER_WORD weight filters packed into the first
        // 63 bits, the last bit is unused. Wts are stored in output-major order.
        // -------------------------------------------------------------------
        LOOP_LOAD_WTS:
        for (IdxType m = 0; m < conv_i; ++m) {
          for (ap_uint<2> kr = 0; kr < K; ++kr) {
            for (ap_uint<2> kc = 0; kc < K; ++kc)
              conv_params[m][kr][kc] = wt_word_buffer[m][kr*K+kc];
          }
        }

        DB(3, print_params(conv_params) );
      }

      // -------------------------------------------------------------------
      // Every word in an image
      // -------------------------------------------------------------------
      // Load word
      if (wrd != words_per_image) {
        LOOP_CONVOLVER_LOAD:
        for (IdxType m = 0; m < conv_i; ++m) {
@@@ bin_conv_dmem_read_seq
          Word word = dmem_i[m][img * words_per_image + wrd];
@@@ bin_conv_dmem_read_pipe
          Word word;
          if (!read_fb)
            word = dmem_i[m][img * words_per_image + wrd];
          else
            word = dmem_fb_i[m][img * words_per_image + wrd];
@@@ bin_conv_5
          for (IdxType bank = 0; bank < CONV_BANKS; ++bank) {
            for (IdxType cc = 0; cc < CONV_COLS-2; ++cc) {
              word_buffer[m][bank][cc+1] = encode_bit(word[ap_uint<6>(bank*BANK_WIDTH+cc)]);
            }
            word_buffer[m][bank][0          ] = (bank==0)            ?
              TwoBit(0) : encode_bit(word[ap_uint<6>(bank*BANK_WIDTH-1)]);
            word_buffer[m][bank][CONV_COLS-1] = (bank==CONV_BANKS-1) ?
              TwoBit(0) : encode_bit(word[ap_uint<6>(bank*BANK_WIDTH+BANK_WIDTH)]);
          }
        }
      }

      // Compute
      LOOP_CONVOLVERS:
      for (IdxType m = 0; m < conv_i; ++m) {
        // Do the following for each word in an image
        process_word( word_buffer[m], old_word_buffer[m], lb, rb, line_buffer[m], conv_params[m],
            conv_out_buffer[m], log_width, words_per_image, wrd );
      } // CONVOLVERS

      for (IdxType m = 0; m < conv_i; ++m) {
        for (IdxType bank = 0; bank < CONV_BANKS; ++bank) {
          for (IdxType cc = 0; cc < CONV_COLS; ++cc) {
            old_word_buffer[m][bank][cc] = word_buffer[m][bank][cc];
          }
        }
      }

      // -------------------------------------------------------------------
      // Sum results across convolvers
      // -------------------------------------------------------------------
      Word binword;
      for (IdxType i = 0; i < WORD_SIZE; ++i) {
        // Ignore conv results after processing the first word
        ConvSum s = img == 0 || wrd == 0 ? ConvSum(0) : fixed_buffer[wrd-1][i];
        for (IdxType m = 0; m < conv_i; ++m)
          s += conv_out_buffer[m][i];
        if (wrd > 0)
          fixed_buffer[wrd-1][i] = s;
        binword[i] = (s >= nc) ? 0 : 1;
      }

      {
        Address o_bank_idx = bank_idx;
        Address o_bank_offset = bank_off*words_per_image + wrd - 1;
        const ap_uint<6> out_offset = ((wrd - 1) % 4) << 4;

        if (norm_mode == 1) {
          if (wrd > 0 && img == images_per_conv - 1)
            outword = binword;
        }
        else if (norm_mode == 2) {
          // horizontal pooling first
          ap_int<WORD_SIZE/2> poolword_h;
          for (ap_uint<6> i = 0; i < WORD_SIZE/2; ++i) {
            poolword_h[i] = binword[2*i] & binword[2*i+1];
          }

          // vertical pooling
          for (ap_uint<6> i = 0; i < WORD_SIZE/4; ++i) {
            // source indices
            ap_uint<5> i0 = i >> (log_width-1);
            i0 = (i0 << log_width) + i(log_width-2,0);
            ap_uint<5> i1 = i0 + pool_width;
            // dest index
            ap_uint<6> d0 = out_offset + i;
            poolword[d0] = poolword_h[i0] & poolword_h[i1];
          }

          // For log_width > 3 we can just assign the word, but log_width = 3 means width = 8,
          // which means pooled width = 4, which is only 16 bits, which is less than 1 Word.
          // So each time we get here we only have 16 bits, meaning we have to accumulate four
          // of these 16-bit batches before writing a word out.
          if (wrd > 0 && img == images_per_conv - 1)
          {
            if (log_width != LOG_BANK_WIDTH)
              outword = poolword;
            else
              outword = (poolword(15, 0), outword(63, 16));
          }

          if (log_width != LOG_BANK_WIDTH) {
            o_bank_offset /= 4;
          } else {
            o_bank_idx = (o_index/4)%conv;
            o_bank_offset = (o_index/4)/conv;
          }
        }

        if (wrd > 0 && img == images_per_conv - 1) {
@@@ bin_conv_dmem_write_seq
          dmem_o[o_bank_idx][o_bank_offset] = outword;
@@@ bin_conv_dmem_write_pipe
          if (!write_fb)
            dmem_o[o_bank_idx][o_bank_offset] = outword;
          else
            dmem_fb_o[o_bank_idx][o_bank_offset] = outword;
@@@ bin_conv_6
        }
      }
    } // wrd
  } // img
}

// -----------------------------------------------------------------------
// Module to do the first conv layer
// -----------------------------------------------------------------------
@@@ fp_conv_hdr_dram
template<int conv_i, int conv_o, int c_wt_words, int kh_words, int c_dmem_i_words, int c_dmem_o_words> void fp_conv(
    Word (& wt_mem)[conv_i][c_wt_words],
    Word (& kh_mem)[kh_words],
@@@ fp_conv_hdr_bram
template<int conv_i, int conv_o, int c_dmem_i_words, int c_dmem_o_words> void fp_conv(
@@@ fp_conv_0
    Word (& dmem_i)[conv_i][c_dmem_i_words],
    Word (& dmem_o)[conv_o][c_dmem_o_words],
    const Address kh_index,
    const Address o_index,
    const unsigned N
) {
  const unsigned M = 3;
  const unsigned S = 32;
  const unsigned OUTWORDS = 16; // words per output image

@@@ fp_conv_arrays
  static unsigned long long wt_mem[conv_i][C_WT_WORDS_0] =
#include "fp_conv_wt.h"
#pragma HLS ARRAY_PARTITION variable=wt_mem complete dim=1

  static long long kh_mem[KH_WORDS_0] =
#include "fp_conv_nc.h"
@@@ fp_conv_1

  C1InputType win[M][K][K];
#pragma HLS ARRAY_PARTITION variable=win complete dim=0
  C1InputType lbuf[M][K-1][S];
#pragma HLS ARRAY_PARTITION variable=lbuf complete dim=0
  Word outwords[OUTWORDS];
#pragma HLS ARRAY_PARTITION variable=outwords complete dim=0
  WtType wtbuf[M];
#pragma HLS ARRAY_PARTITION variable=wtbuf complete dim=0

  Word outword;

  Address wt_offset = 0;
  ap_uint<3> wt_addr = 0;

#ifndef __SYNTHESIS__
  for (int m = 0; m < M; ++m)
    for (int r = 0; r < K; ++r)
      for (int c = 0; c < K; ++c)
        win[m][r][c] = 0;

  for (int m = 0; m < M; ++m)
    for (int r = 0; r < K-1; ++r)
      for (int c = 0; c < S; ++c)
        lbuf[m][r][c] = 0;
#endif

  // Parallelized across m, better for HLS
  LOOP_FP_CONV_O:
  for (IdxType n = 0; n < N; ++n) {
@@@ fp_conv_2
#pragma HLS loop_tripcount max={}
@@@ fp_conv_3

    // The weights for the 1st conv layer are just laid out
    // linearly across wt_mem, 3 weights per 64-bit word
    DB_PRINT(3, "n = %u\n", n.to_int());
    Word wt_word = wt_mem[n % conv_i][n / conv_i];
    LOOP_LOAD_WTS:
    for (ap_uint<2> m = 0; m < M; ++m) {
#pragma HLS UNROLL
      wtbuf[m] = wt_word((m+1)*WT_SIZE-1, m*WT_SIZE);
      DB(3, print_wt(wtbuf[m]));
      DB(3, printf("--\n"));
    }

    // load batch norm params
    C1Comp nc;
    load_kh(nc, kh_mem, (kh_index+n));
    //printf ("  n=%3d, nc=%6.3f\n", n.to_int(), nc.to_float());

    // begin convolution
    LOOP_CONV_ROWS: for (IdxType r = 0; r < S+1; ++r) {
      if (r % 2 == 1) {
        outword = 0;
      }
      LOOP_CONV_COLS: for (IdxType c1 = 0; c1 < S+1; c1 += conv_i) {
#pragma HLS PIPELINE
        LOOP_CONV_CONV: for (IdxType c2 = 0; c2 < conv_i; c2++) {
          IdxType c = c1 + c2;

          // load input word
          const Address addr = r*S + c;
          Word inword = r < S && c < S ? dmem_i[addr%conv_i][addr/conv_i] : Word(0);
           
          for (ap_uint<2> m = 0; m < M; ++m) {
            // load data: the value of pix is either the pixel at [r,c]
            // 0 -> +1, -1 -> -1
            // or -> 0 for padding around the boundaries
            C1InputType pix;
            const unsigned W = pix.length();
            pix(W-1,0) = inword(W-1+m*W, m*W);

            // window: shift left, leaving rightmost col for new data
            for (IdxType wr = 0; wr < K; ++wr) {
              for (IdxType wc = 0; wc < K-1; ++wc) {
                win[m][wr][wc] = win[m][wr][wc+1];
              }
            }

            // window: fill top K-1 pixels of rightmost column from lbuf
            for (IdxType wr = 0; wr < K-1; ++wr) {
              C1InputType val = (c < S && r + wr >= K - 1) ? lbuf[m][wr][c] : C1InputType(0);
              win[m][wr][K-1] = val;
            }

            // window: fill bottom right with new input pixel
            win[m][K-1][K-1] = pix;

            // lbuf: shift up column c
            if (c < S) {
              for (IdxType lr = 0; lr < K-2; ++lr) {
                lbuf[m][lr][c] = lbuf[m][lr+1][c];
              }
              lbuf[m][K-2][c] = pix;
            }
          } // m

          // only perform the conv and store if legal position
          C1ConvType res = 0;
          for (ap_uint<2> m = 0; m < M; ++m) {
            for (ap_uint<2> wr = 0; wr < K; ++wr) {
              for (ap_uint<2> wc = 0; wc < K; ++wc) {
                const C1InputType& pix = win[m][wr][wc];
                const Bit& b = wtbuf[m][8-(wr*K+wc)];
                res += (b==0) ? pix : (C1InputType)(-pix);
              } 
            }
          }

          if (r > 0 && c > 0 && c < S + 1) {
            // perform normalization right here
            outword[((r-1)%2)*S + c-1] = (res >= nc) ? Bit(0) : Bit(-1);
          }
        }
      } // CONV_COLS
      if (r > 0 && r % 2 == 0)
        outwords[(r-1)/2] = outword;
    } // CONV_ROWS

    // Here i is the word offset within the outwords buffer
    LOOP_OUTPUT:
    for (IdxType i = 0; i < OUTWORDS; ++i) {
#pragma HLS PIPELINE
      Address img_idx = o_index+n;
      Address bank_idx = img_idx % conv_o;
      Address bank_off = img_idx / conv_o;
      dmem_o[bank_idx][bank_off*OUTWORDS + i] = outwords[i];
    }
  } // n
}

@@@ bin_dense_hdr_dram
template<int conv_i, int conv_o, int c_wt_words, int kh_words, int c_dmem_i_words, int c_dmem_o_words> void bin_dense(
    const Word (& wt_mem)[conv_i][c_wt_words],
    const Word (& kh_mem)[kh_words],
@@@ bin_dense_hdr_bram
template<int conv_i, int conv_o, int c_dmem_i_words, int c_dmem_o_words> void bin_dense(
@@@ bin_dense_hdr_pipe
template<int conv_i, int conv_o, int c_dmem_i_words, int c_dmem_o_words, int c_dmem_fb_words> void bin_dense_{}(
@@@ bin_dense_0
    Word (& dmem_i)[conv_i][c_dmem_i_words],
    Word (& dmem_o)[conv_o][c_dmem_o_words],
@@@ bin_dense_hdr_fb
    Word (& dmem_fb_i)[conv_i][c_dmem_fb_words],
    Word (& dmem_fb_o)[conv_i][c_dmem_fb_words],
    const ap_uint<1> read_fb,
    const ap_uint<1> write_fb,
@@@ bin_dense_1
    ap_uint<2> layer_type,
    const Address o_index,
    const unsigned n_inputs,
    const unsigned n_outputs,
    int layer
) {
  //assert(n_outputs % WORD_SIZE == 0);
  assert(layer_type == LAYER_DENSE || n_outputs == 10);
  assert(n_inputs/WORD_SIZE % conv_i == 0);

@@@ bin_dense_wt_mem
  static long long wt_mem_{}[{}][{}] =
#include "bin_dense_{}_wt.h"
@@@ bin_dense_pragma
#pragma HLS ARRAY_PARTITION variable=wt_mem_{} complete dim=1
@@@ bin_dense_kh_mem
  static long long kh_mem_0[KH_WORDS_6] =
#include "bin_dense_0_nc.h"
  static long long kh_mem_1[KH_WORDS_7] =
#include "bin_dense_1_nc.h"
  static long long kh_mem_2[KH_WORDS_8] =
#include "bin_dense_2_nc.h"

@@@ bin_dense_2
  DenseSum sum_m[conv_i];
  // for last layer
  DenseNorm best_out = -1024;
  ap_int<8> prediction = -1;
  Word o_word = 0;

  // read words from dmem and the wt store, dot them
  // o is the output bit, i is the input bit
  LOOP_DENSE_O:
  for (Address o = 0; o < n_outputs; ++o) {
    const Address o_addr = (o_index+o)/WORD_SIZE;
    const ap_uint<6> o_offset = (o_index+o) % WORD_SIZE;
@@@ bin_dense_dmem_o_read
    o_word = dmem_o[o_addr%conv_o][o_addr/conv_o];
@@@ bin_dense_3

    DenseSum sum = 0;

    LOOP_DENSE_I:
    for (Address i = 0; i < n_inputs; i+=conv_i*WORD_SIZE) {{
#pragma HLS loop_tripcount max={}
#pragma HLS PIPELINE
      const Address wt_addr = (o*n_inputs+i) / WORD_SIZE;

      for (IdxType j = 0; j < conv_i; ++j) {{
// HG: The following pragma may seem redundant, but Vivado HLS optimizes the
// pipeline pragma in the surrounding loop away when the number of loop
// iterations is 1.
#pragma HLS UNROLL
        // in_wrd addr = [(i/WORD_SIZE+j) % CONVOLVERS][(i/WORD_SIZE+j) / CONVOLVERS]
        // wt_wrd addr = [wt_addr % CONVOLVERS][wt_addr / CONVOLVERS]
@@@ bin_dense_dmem_i_read_seq
        const Word in_wrd = dmem_i[j][i/WORD_SIZE/conv_i];
@@@ bin_dense_dmem_i_read_pipe
        Word in_wrd;
        if (!read_fb)
          in_wrd = dmem_i[j][i/WORD_SIZE/conv_i];
        else
          in_wrd = dmem_fb_i[j][i/WORD_SIZE/conv_i];
@@@ bin_dense_4
        Word wt_wrd;
@@@ bin_dense_wt_read_dram
        wt_wrd = wt_mem[j][wt_addr / conv_i];
@@@ bin_dense_wt_read_bram
        if (layer == {})
          wt_wrd = wt_mem_{}[j][wt_addr / conv_i];
@@@ bin_dense_5

        Word x = wt_wrd ^ in_wrd;

        // count_set bit for 64 bits, returns 2*cnt
        x -= (x >> 1) & m1;
        x = (x & m2) + ((x >> 2) & m2);
        x = (x + (x >> 4)) & m4;
        x += x >> 8;
        x += x >> 16;
        x += x >> 32;
        x = x & 0x7f;

        sum_m[j] = WORD_SIZE - (DenseSum)(x<<1);
      }

      for (IdxType j = 0; j < conv_i; ++j)
        sum += sum_m[j];
    } // n_inputs

    // not last layer -> biniarize,
    // otherwise just store the value as a 64bit word
    Address kh_addr = o / (layer_type == LAYER_DENSE ? KH_PER_WORD : 2);
@@@ bin_dense_kh_read_dram
    Word kh_word = kh_mem[kh_addr];
@@@ bin_dense_kh_read_bram
    Word kh_word;
    if (layer == 0)
      kh_word = kh_mem_0[kh_addr];
    else if (layer == 1)
      kh_word = kh_mem_1[kh_addr];
    else if (layer == 2)
      kh_word = kh_mem_2[kh_addr];
@@@ bin_dense_6

    if (layer_type == LAYER_DENSE) {
      NormComp nc;
      IdxType kh_off = o % KH_PER_WORD;
      if (kh_off == 0)
        nc(15,0) = kh_word(15, 0);
      else if (kh_off == 1)
        nc(15,0) = kh_word(31,16);
      else if (kh_off == 2)
        nc(15,0) = kh_word(47,32);
      else
        nc(15,0) = kh_word(63,48);

      o_word[o_offset] = (sum >= nc) ? 0 : 1;
    } else {
      KType ki;  HType hi;
      IdxType kh_off = o % 2;
      if (kh_off == 0) {
        ki(15,0) = kh_word(15, 0);
        hi(15,0) = kh_word(31,16);
      } else {
        ki(15,0) = kh_word(47,32);
        hi(15,0) = kh_word(63,48);
      }

      //printf (" >> %d * %f + %f\n", sum.to_int(), ki.to_float(), hi.to_float());
      ap_fixed<20,10> out = ap_fixed<20,10>(sum)*ki + hi;

      if (o == 0 || out > best_out) {
        prediction = o;
        best_out = out;
      }
    }

@@@ bin_dense_dmem_write_seq
    dmem_o[o_addr%conv_o][o_addr/conv_o] = o_word;
@@@ bin_dense_dmem_write_pipe
    if (o % WORD_SIZE == WORD_SIZE - 1 || o == n_outputs - 1) {
      if (write_fb)
        dmem_fb_o[o_addr%conv_i][o_addr/conv_i] = o_word;
      else
        dmem_o[o_addr%conv_o][o_addr/conv_o] = o_word;
    }
@@@ bin_dense_7
  } // n_outputs

  // Here we are using o_index as a bit index, not a word index!
  if (layer_type == LAYER_LAST) {
    Word o_word;
    o_word(7,0) = prediction(7,0);
    o_word(WORD_SIZE-1, 8) = 0;
    dmem_o[0][0] = o_word;
  }
}

@@@ top_seq_0
void top(
    int layer,
@@@ top_hdr_wts
    Word wt_i[WT_WORDS],
    Word kh_i[KH_WORDS],
    const Address wt_words,
    const Address kh_words,
@@@ top_seq_1
    Word dmem_i[DMEM_WORDS],
    Word dmem_o[DMEM_WORDS],
    const Address    n_inputs,
    const Address    n_outputs,
    const Address    input_words,
    const Address    output_words,
    const ap_uint<3> layer_mode,  // [0]='new layer', [2:1]='conv1,conv,dense,last'
    const ap_uint<1> dmem_mode,   // 0 means dmem[0] is input
    const ap_uint<2> width_mode,  // 0=8'b, 1=16'b, 2=32'b
    const ap_uint<2> norm_mode    // 0='do nothing', 1='do norm', 2='do pool'
) {
  DB_PRINT(2, "==== Entering Accel ====\n");
  const ap_uint<2> layer_type = layer_mode(2,1);
  const unsigned width = 8 << width_mode;
  DB_PRINT(1, "  Inputs  = %d\n", n_inputs.to_int());
  DB_PRINT(1, "  Outputs = %d\n", n_outputs.to_int());
  DB_PRINT(1, "  i_words = %d\n", input_words.to_int());
  DB_PRINT(1, "  o_words = %d\n", output_words.to_int());
  DB_PRINT(1, "  Width = %d\n", width);
  DB_PRINT(1, "  layer_mode = %d %d\n", layer_mode[0]==0 ? 0 : 1, layer_type.to_int());
  DB_PRINT(1, "  dmem_mode = %d\n", dmem_mode.to_int());

  assert(width <= MAX_WIDTH);
  assert(n_inputs != 0);
  if (layer_type <= LAYER_CONV) {
    assert(input_words % CONVOLVERS == 0);
    assert(n_inputs*width*width <= DMEM_WORDS*WORD_SIZE);
    assert(n_inputs*WT_SIZE <= WT_WORDS*WORD_SIZE);
  }

  static Word dmem[2][CONVOLVERS][C_DMEM_WORDS];
#pragma HLS ARRAY_PARTITION variable=dmem complete dim=1
#pragma HLS ARRAY_PARTITION variable=dmem complete dim=2
@@@ top_wt_arrays_dram
  static Word kh_mem[KH_WORDS];
  static Word wt_mem[CONVOLVERS][C_WT_WORDS];
#pragma HLS ARRAY_PARTITION variable=wt_mem complete dim=1
@@@ top_wt_arrays_bram
  static Word kh_mem_1[KH_WORDS_1] =
#include "bin_conv_0_nc.h"
  static Word kh_mem_2[KH_WORDS_2] =
#include "bin_conv_1_nc.h"
  static Word kh_mem_3[KH_WORDS_3] =
#include "bin_conv_2_nc.h"
  static Word kh_mem_4[KH_WORDS_4] =
#include "bin_conv_3_nc.h"
  static Word kh_mem_5[KH_WORDS_5] =
#include "bin_conv_4_nc.h"
@@@ top_seq_2
  static Address kh_index = 0;
  static Address o_index = 0;

  if (layer_mode[0]) {
    kh_index = 0;
    o_index = 0;
  } else {
    kh_index = kh_index[0];
  }

  ap_uint<1> d_i_idx = dmem_mode;
  ap_uint<1> d_o_idx = ~dmem_mode;

  // Data input
  const ap_uint<5> words_per_image = 1 << (2*width_mode);
  Address img_idx = 0;  // i / words_per_image;
  IdxType img_off = 0;  // i % words_per_image;
  LOOP_DMEM_I: for (Address i = 0; i < input_words; ++i) {
#pragma HLS loop_tripcount max=2048
#pragma HLS PIPELINE
    if (layer_type == LAYER_CONV) {
      Address bank_idx = img_idx % CONVOLVERS;
      Address bank_off = img_idx / CONVOLVERS;
      dmem[d_i_idx][bank_idx][(bank_off<<(2*width_mode)) + img_off] = dmem_i[i];
    }
    else
      dmem[d_i_idx][i%CONVOLVERS][i/CONVOLVERS] = dmem_i[i];

    if (++img_off == words_per_image) {
      img_off = 0;
      ++img_idx;
    }
  }

#ifndef __SYNTHESIS__
  if (o_index == 0)
    print_dmem(dmem[d_i_idx], layer);
#endif

@@@ top_wt_load
  // Weight input, we must copy every 64-bit Word from the interface
  // into the accelerator
  LOOP_WT_I: for (Address i = 0; i < wt_words; ++i) {
#pragma HLS loop_tripcount max=131072
#pragma HLS PIPELINE
    wt_mem[i%CONVOLVERS][i/CONVOLVERS] = wt_i[i];
  }
  //printf ("\nAccel Weights:\n");
  //print_params3d(wt_mem[0], 0, n_inputs*n_outputs);

  LOOP_KH_I: for (ap_uint<16> i = 0; i < kh_words; ++i)
#pragma HLS loop_tripcount max=256
#pragma HLS PIPELINE
    kh_mem[i] = kh_i[i];

@@@ top_seq_3
  if (layer_type == LAYER_CONV1) {
    assert(n_inputs == 3);

    fp_conv(
@@@ pass_fp_conv_wt
        wt_mem,
        kh_mem,
@@@ top_seq_4
        dmem[d_i_idx],
        dmem[d_o_idx],
        kh_index,
        o_index,
        n_outputs
    );

  }
  else if (layer_type == LAYER_CONV) {
    assert(norm_mode != 2 || n_outputs % 4 == 0); // needed for pooling of 8x8 image
    assert(n_inputs % CONVOLVERS == 0);

    LOOP_IMG_BATCH:
    for (IdxType i = 0; i < n_outputs; ++i) {
#pragma HLS loop_tripcount max=512

      // Load the batch-norm parameters for this output
      NormComp nc;
@@@ top_kh_load_dram
      load_kh(nc, kh_mem, kh_index + i);
@@@ top_kh_load_bram
      if (layer == 1)
        load_kh(nc, kh_mem_1, kh_index + i);
      else if (layer == 2)
        load_kh(nc, kh_mem_2, kh_index + i);
      else if (layer == 3)
        load_kh(nc, kh_mem_3, kh_index + i);
      else if (layer == 4)
        load_kh(nc, kh_mem_4, kh_index + i);
      else if (layer == 5)
        load_kh(nc, kh_mem_5, kh_index + i);
@@@ top_seq_5

      if (d_i_idx == 0)
        bin_conv(
@@@ pass_bin_conv_wt_1
            wt_mem,
@@@ top_seq_6
            nc,
            dmem[0],
            dmem[1],
            n_inputs,
            o_index + i,
            i == 0 ? 1 : 0,         // new_batch
            width_mode,
            norm_mode,
            layer - 1
        );
      else
        bin_conv(
@@@ pass_bin_conv_wt_2
            wt_mem,
@@@ top_seq_7
            nc,
            dmem[1],
            dmem[0],
            n_inputs,
            o_index + i,
            i == 0 ? 1 : 0,         // new_batch
            width_mode,
            norm_mode,
            layer - 1
        );
    }
  }
  else {
    bin_dense(
@@@ pass_bin_dense_wt
        wt_mem,
        kh_mem,
@@@ top_seq_8
        dmem[d_i_idx],
        dmem[d_o_idx],
        layer_type,
        o_index,
        n_inputs, n_outputs,
        layer - 6
    );
  } // layer_type
  
#ifndef __SYNTHESIS__
  if (output_words > 0)
    print_dmem(dmem[d_o_idx], layer + 1);
#endif

  // Data output
  ap_uint<5> words_per_out = words_per_image / ((norm_mode!=2) ? 1 : 4);
  img_idx = 0;
  img_off = 0;
  LOOP_DMEM_O: for (Address i = 0; i < output_words; ++i) {
#pragma HLS loop_tripcount max=2048

#pragma HLS PIPELINE
    // exclude conv6 (width==8, norm_mode==2) here because it writes
    // the output fmaps linearly
    if (layer < 5) {
      Address bank_idx = img_idx % CONVOLVERS;
      Address bank_off = img_idx / CONVOLVERS;
      dmem_o[i] = dmem[d_o_idx][bank_idx][bank_off*words_per_out + img_off];
    }
    else
      dmem_o[i] = dmem[d_o_idx][i%CONVOLVERS][i/CONVOLVERS];

    if (++img_off == words_per_out) {
      img_off = 0;
      ++img_idx;
    }
  }

  kh_index += n_outputs;
  o_index += n_outputs;
}
@@@ multi_bin_conv_hdr
template <int conv_i, int conv_o, int c_dmem_i_words, int c_dmem_o_words> void multi_bin_conv_{}(
@@@ multi_bin_conv_0
    Word (& dmem_i)[conv_i][c_dmem_i_words],
    Word (& dmem_o)[conv_o][c_dmem_o_words]
) {
@@@ multi_bin_conv_arrays
  static Word kh_mem_{}[{}] =
#include "bin_conv_{}_nc.h"
@@@ multi_bin_conv_dmem
  
  Word dmem[2][conv_i][{}];
#pragma HLS ARRAY_PARTITION variable=dmem complete dim=1
#pragma HLS ARRAY_PARTITION variable=dmem complete dim=2

@@@ multi_bin_conv_layers
  int first = {};
  int last = {};
  LOOP_LAYERS: for (int layer = first; layer <= last; layer++)
@@@ multi_bin_conv_1
  {
#ifndef __SYNTHESIS__
    if (layer == first)
      print_dmem(dmem_i, layer + 1);
    else
      print_dmem(dmem[layer % 2], layer + 1);
#endif

    Address n_inputs = 128 << (layer / 2);
    Address n_outputs = 128 << ((layer + 1) / 2);
    ap_uint<2> norm_mode = 2 - (layer & 1);
    unsigned width_mode = 2 - ((layer + 1) / 2);
    
    assert(norm_mode != 2 || n_outputs % 4 == 0); // needed for pooling of 8x8 image
    assert(n_inputs % conv_i == 0);

    LOOP_IMG_BATCH:
    for (IdxType i = 0; i < n_outputs; ++i) {

      // Load the batch-norm parameters for this output
      NormComp nc;
@@@ multi_bin_conv_kh_read
      if (layer == {})
        load_kh(nc, kh_mem_{}, i);
@@@ multi_bin_conv_call

      bin_conv_{}(
          nc,
          dmem_i,
          dmem_o,
          dmem[layer % 2],
          dmem[(layer + 1) % 2],
          layer != first,
          layer != last,
          n_inputs,
          i,
          i == 0 ? 1 : 0,         // new_batch
          width_mode,
          norm_mode,
          layer
      );
@@@ multi_bin_conv_2
    }
  }
}

@@@ multi_bin_dense_hdr
template <int conv_i, int conv_o, int c_dmem_i_words, int c_dmem_o_words> void multi_bin_dense_{}(
@@@ multi_bin_dense_0
    Word (& dmem_i)[conv_i][c_dmem_i_words],
    Word (& dmem_o)[conv_o][c_dmem_o_words]
) {
@@@ multi_bin_dense_dmem
  Word dmem[2][conv_i][{}];
#pragma HLS ARRAY_PARTITION variable=dmem complete dim=1
#pragma HLS ARRAY_PARTITION variable=dmem complete dim=2

@@@ multi_bin_dense_layers
  int first = {};
  int last = {};
  LOOP_LAYERS: for (int layer = first; layer <= last; layer++)
@@@ multi_bin_dense_1
  {{
#pragma HLS loop_tripcount max={}
#ifndef __SYNTHESIS__
    if (layer == first)
      print_dmem(dmem_i, layer + 6);
    else
      print_dmem(dmem[layer % 2], layer + 6);
#endif

    Address n_inputs = layer == 0 ? 8192 : 1024;
    Address n_outputs = layer == 2 ? 10 : 1024;
    
@@@ multi_bin_dense_call
    bin_dense_{}(
        dmem_i,
        dmem_o,
        dmem[layer % 2],
        dmem[(layer + 1) % 2],
        layer != first,
        layer != last,
        layer == 2 ? LAYER_LAST : LAYER_DENSE,
        0,
        n_inputs, n_outputs,
        layer
    );
@@@ multi_bin_dense_2
  }
}

@@@ top_pipe_0
void top(
    Word dmem_i[DMEM_WORDS_0],
    ap_uint<4> & pred
) {
@@@ top_pipe_dmem
  Word dmem_{}[{}][{}];
#pragma HLS ARRAY_PARTITION variable=dmem_{} complete dim=1
@@@ top_pipe_1

#pragma HLS dataflow

  LOOP_DMEM_I: for (Address i = 0; i < DMEM_WORDS_0; ++i) {
#pragma HLS PIPELINE
    dmem_0[i%CONVOLVERS_0][i/CONVOLVERS_0] = dmem_i[i];
  }

#ifndef __SYNTHESIS__
  print_dmem(dmem_0, 0);
#endif

  fp_conv(
      dmem_0,
      dmem_1,
      0,
      0,
      128
  );

@@@ top_pipe_bin_conv
#ifndef __SYNTHESIS__
  print_dmem(dmem_{}, {});
#endif

  static Word kh_mem_{}[{}] =
#include "bin_conv_{}_nc.h"
  
  LOOP_IMG_BATCH_{}:
  for (IdxType i = 0; i < {}; ++i) {{

    // Load the batch-norm parameters for this output
    NormComp nc;
    load_kh(nc, kh_mem_{}, i);

    bin_conv_{}(
        nc,
        dmem_{},
        dmem_{},
        dmem_{},
        dmem_{},
        false,
        false,
        {},
        i,
        i == 0 ? 1 : 0,         // new_batch
        {},
        {},
        {}
    );
  }}
@@@ top_pipe_multi_bin_conv
  multi_bin_conv_{}(dmem_{}, dmem_{});
@@@ top_pipe_2

@@@ top_pipe_bin_dense
#ifndef __SYNTHESIS__
  print_dmem(dmem_{}, {});
#endif

  bin_dense_{}(
      dmem_{},
      dmem_{},
      dmem_{},
      dmem_{},
      false,
      false,
      {},
      0,
      {},
      {},
      {}
  );
@@@ top_pipe_multi_bin_dense
  multi_bin_dense_{}(dmem_{}, dmem_{});
@@@ top_pipe_3
  
#ifndef __SYNTHESIS__
  print_dmem(dmem_{}, 9);
#endif

  pred = dmem_{}[0][0](3, 0);
@@@ top_pipe_4
}
