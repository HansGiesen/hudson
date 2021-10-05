@@@ bnn_0
#define IMAGE_COUNT        (100)
#define DESIRED_ERROR_RATE (0.12)

#include <cstddef>
#include <cstdlib>

#include "Accel.h"
#include "AccelSchedule.h"
#include "AccelTest.h"
#include "Dense.h"
#ifndef RUN_STANDALONE
#include "ZipIO.h"
#endif
#include "ParamIO.h"
#include "DataIO.h"
#include "Timer.h"

#ifdef __SDSCC__
#include <sds_lib.h>
#endif

#ifdef RUN_STANDALONE
#include <ff.h>
#endif

#include <utility>

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <Test directory> <Number of images>\n", argv[0]);
    return 0;
  }
  const unsigned n_imgs = std::stoi(argv[2]);
  const std::string test_dir(argv[1]);

  const unsigned lconv  = 6;  // last conv
  const unsigned ldense = 8;  // last dense
  const bool DENSE_LAYER_CPU = getenv("BNN_DENSE_LAYER_CPU") != NULL;
  const bool LAST_LAYER_CPU = getenv("BNN_LAST_LAYER_CPU") != NULL;
  if (DENSE_LAYER_CPU)
    printf ("## Dense layer CPU is turned on ##\n");
  if (LAST_LAYER_CPU)
    printf ("## Last layer CPU is turned on ##\n");

  // print some config numbers
  printf ("* WT_WORDS   = %u\n", WT_WORDS);
  printf ("* KH_WORDS   = %u\n", KH_WORDS);

  // Load input data
  printf ("## Loading input data ##\n");
  Cifar10TestInputs X(test_dir + "/data/cifar10_test_inputs.zip", n_imgs);
  Cifar10TestLabels y(test_dir + "/data/cifar10_test_labels.zip", n_imgs);

  // Load parameters
  printf ("## Loading parameters ##\n");
  Params params(test_dir + "/params/cifar10_parameters_nb.zip");

  // ---------------------------------------------------------------------
  // allocate and binarize all weights
  // ---------------------------------------------------------------------
  Word* wt[N_LAYERS];
  Word* kh[N_LAYERS];
  for (unsigned l = 0; l < N_LAYERS; ++l) {
    const unsigned M = M_tab[l];
    const unsigned N = N_tab[l];
    if (layer_is_conv(l+1))
      wt[l] = new Word[WTS_TO_WORDS(M*N)];
    else
      wt[l] = new Word[M*N / WORD_SIZE];
    const float* weights = params.float_data(widx_tab[l]);
    set_weight_array(wt[l], weights, l+1);

    kh[l] = new Word[N/KH_PER_WORD * sizeof(Word)];
    const float* k = params.float_data(kidx_tab[l]);
    const float* h = params.float_data(hidx_tab[l]);
    set_bnorm_array(kh[l], k, h, l+1);
  }

@@@ bnn_comp_sched
  // ---------------------------------------------------------------------
  // // compute accelerator schedule (divides up weights)
  // ---------------------------------------------------------------------
  AccelSchedule layer_sched[N_LAYERS];
  for (unsigned l = 0; l < N_LAYERS; ++l) {
    compute_accel_schedule(
        wt[l], kh[l],
        M_tab[l], N_tab[l], S_tab[l], T_tab[l], pool_tab[l],
        layer_sched[l]
    );
  }

  // allocate memories for data i/o for the accelerator
  Word* data_i  = (Word*) MEM_ALLOC( BATCH_SIZE * DMEM_WORDS * sizeof(Word) );
  Word* data_o  = (Word*) MEM_ALLOC( BATCH_SIZE * DMEM_WORDS * sizeof(Word) );
  if (!data_i || !data_o) {
    fprintf (stderr, "**** ERROR: Alloc failed in %s\n", __FILE__);
    return (-2);
  }

  const unsigned n_batches = (n_imgs + BATCH_SIZE - 1) / BATCH_SIZE;

@@@ bnn_dmem_alloc
  // allocate memories for data i/o for the accelerator
  Word* data_i[N_LAYERS];
  for (int n = 0; n < N_LAYERS; n++) {
    data_i[n] = (Word*) MEM_ALLOC( DMEM_WORDS_0 * sizeof(Word) );
    if (!data_i[n]) {
      fprintf (stderr, "**** ERROR: Alloc failed in %s\n", __FILE__);
      return (-2);
    }
  }

@@@ bnn_1
  //--------------------------------------------------------------
  // Run BNN
  //--------------------------------------------------------------

  unsigned n_errors = 0;
  printf ("## Running BNN for %d images\n", n_imgs);

#ifdef __SDSCC__
  unsigned long long Start_time_HW = sds_clock_counter();
#endif
@@@ bnn_seq
  for (unsigned n = 0; n < n_batches; ++n) {
    float* data = X.data + n*BATCH_SIZE*3*32*32;
    const unsigned img_cnt = n == n_batches - 1 ? n_imgs - n * BATCH_SIZE : BATCH_SIZE;
    binarize_input_images(data_i, data, 32, img_cnt);

    //------------------------------------------------------------
    // Execute conv layers
    //------------------------------------------------------------
    for (unsigned l = 1; l <= lconv; ++l) {
      const unsigned M = M_tab[l-1];
      const unsigned N = N_tab[l-1];
      const unsigned S = S_tab[l-1];
      unsigned input_words = (l==1) ? S*S : M*S*S/WORD_SIZE;
      unsigned output_words = (pool_tab[l-1]) ? N*S*S/WORD_SIZE/4 : N*S*S/WORD_SIZE;

      run_accel_schedule(
          data_i, data_o,
          l-1,        // layer_idx
          (l==1 || BATCH_SIZE > 1) ? input_words : 0,
          (l==lconv && DENSE_LAYER_CPU || BATCH_SIZE > 1) ? output_words : 0,
          img_cnt,
          l % 2,      // mem_mode
          layer_sched[l-1]
      );

      if (BATCH_SIZE > 1)
        std::swap(data_i, data_o);
    }

    //------------------------------------------------------------
    // Execute dense layers
    //------------------------------------------------------------
    for (unsigned l = lconv+1; l <= ldense; ++l) {
      const unsigned M = M_tab[l-1];
      const unsigned N = N_tab[l-1];

      if (DENSE_LAYER_CPU) {
        for (unsigned i = 0; i < M/WORD_SIZE; ++i)
          data_i[i] = data_o[i];

        dense_layer_cpu(
            wt[l-1], params.float_data(3*l-2), params.float_data(3*l-1),
            data_i, data_o, M, N
        );

      } else {
        run_accel_schedule(
            data_i, data_o,
            l-1,
            BATCH_SIZE > 1 ? M / WORD_SIZE : 0,    // input_words
            (l==ldense && LAST_LAYER_CPU || BATCH_SIZE > 1) ? 1024/WORD_SIZE : 0,
            img_cnt,
            l % 2,
            layer_sched[l-1]
        );
      }

      if (BATCH_SIZE > 1)
        std::swap(data_i, data_o);
    }

    //------------------------------------------------------------
    // Execute last layer
    //------------------------------------------------------------
    if (DENSE_LAYER_CPU || LAST_LAYER_CPU) {
      last_layer_cpu(
          wt[ldense],
          params.float_data(kidx_tab[ldense]),
          params.float_data(hidx_tab[ldense]),
          data_o,
          M_tab[ldense], N_tab[ldense]
      );
    } else {
      run_accel_schedule(
          data_i, data_o,
          ldense,
          BATCH_SIZE > 1 ? M_tab[ldense] : 0,
          BATCH_SIZE > 1 ? N_tab[ldense] : 1,
          img_cnt,
          1,
          layer_sched[ldense]
      );
    }

    for (int img = 0; img < img_cnt; img++) {
      ap_int<8> p = 0;
      p(7,0) = data_o[img](7,0);
      int prediction = p.to_int();

      //assert(prediction >= 0 && prediction <= 9,);
      int label = y.data[n * BATCH_SIZE + img];

      printf ("  Pred/Label:\t%2u/%2d\t[%s]\n", prediction, label,
          ((prediction==label)?" OK ":"FAIL"));

      n_errors += (prediction!=label);
    }
  }
@@@ bnn_pipe
  ap_uint<4> predictions[N_LAYERS];
  for (unsigned n = 0; n < n_imgs + N_LAYERS; ++n) {
    if (n >= N_LAYERS) {
#pragma SDS wait(1)
      int prediction = predictions[n % N_LAYERS].to_int();
      //assert(prediction >= 0 && prediction <= 9);
      int label = y.data[n - N_LAYERS];

      printf ("  Pred/Label:\t%2u/%2d\t[%s]\n", prediction, label,
          ((prediction==label)?" OK ":"FAIL"));

      n_errors += (prediction!=label);
    }

    if (n < n_imgs) {
      float* data = X.data + n*3*32*32;
      binarize_input_images(data_i[n % N_LAYERS], data, 32, 1);

#pragma SDS async(1)
      top(data_i[n % N_LAYERS], predictions[n % N_LAYERS]);
    }
  }
@@@ bnn_2
#ifdef __SDSCC__
  unsigned long long End_time_HW = sds_clock_counter();
#endif

  double error_rate = (1.0 * n_errors) / n_imgs;

  printf ("\n");
  printf ("Errors: %u (%4.2f%%)\n", n_errors, 100.0 * error_rate);
  printf ("\n");
  printf ("Total accel runtime = %10.4f seconds\n", total_time());
  printf ("\n");

#ifdef __SDSCC__
  unsigned long long Duration_HW = End_time_HW - Start_time_HW;
  printf("The hardware test took %llu cycles.\n", Duration_HW);
#endif

  if (error_rate <= DESIRED_ERROR_RATE){
    printf("TEST PASSED\n");
    return 0;
  }
  else{
    printf("TEST FAILED\n");
    return 1;
  }

@@@ bnn_free_seq
  MEM_FREE( data_o );
  MEM_FREE( data_i );
  for (unsigned n = 0; n < N_LAYERS; ++n) {
@@@ bnn_free_pipe
  for (unsigned n = 0; n < N_LAYERS; ++n) {
    MEM_FREE( data_i[n] );
@@@ bnn_3
    delete[] wt[n];
    delete[] kh[n];
  }
  return 0;
}
