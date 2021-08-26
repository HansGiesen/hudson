/*===============================================================*/
/*                                                               */
/*                          digitrec.cpp                         */
/*                                                               */
/*             Hardware function for digit recognition           */
/*                                                               */
/*===============================================================*/

#include "../host/typedefs.h"
#include "sorting_network.hpp"

// popcount function
int popcount(WholeDigitType x)
{
  // most straightforward implementation
  // actually not bad on FPGA
  int cnt = 0;
  for (int i = 0; i < 256; i ++ )
    cnt = cnt + x[i];

  return cnt;
}

// Given the test instance and a (new) training instance, this
// function maintains/updates an array of K minimum
// distances per training set.
void update_knn( bool init, WholeDigitType test_inst,
		         WholeDigitType train_inst, int min_distances[K_CONST] )
{
  #pragma HLS inline

  // Compute the difference using XOR
  WholeDigitType diff = test_inst ^ train_inst;

  int dist = 0;

  dist = popcount(diff);

  bool lt[K_CONST];
#pragma HLS array_partition variable=lt complete dim=1
  for (int i = 0; i < K_CONST; i++)
    lt[i] = dist < min_distances[i];

  int new_distances[K_CONST];
#pragma HLS array_partition variable=new_distances complete dim=1
  for (int i = 0; i < K_CONST; i++)
    if (init && i == 0)
      new_distances[i] = dist;
    else if (init && i > 0)
      new_distances[i] = 255;
    else if (i > 0 && lt[i - 1])
      new_distances[i] = min_distances[i - 1];
    else if (lt[i])
      new_distances[i] = dist;
    else
      new_distances[i] = min_distances[i];

  for (int i = 0; i < K_CONST; i++)
	min_distances[i] = new_distances[i];
}

// Given 10xK minimum distance values, this function
// finds the actual K nearest neighbors and determines the
// final output based on the most common int represented by
// these nearest neighbors (i.e., a vote among KNNs).
void knn_vote_insert_sort( int knn_set[PIPELINE_CNT][K_CONST], LabelType & result )
{
  #pragma HLS inline

  // local buffers

  // final K nearest neighbors
  int min_distance_list[K_CONST];
  #pragma HLS array_partition variable=min_distance_list complete dim=0
  // labels for the K nearest neighbors
  int label_list[K_CONST];
  #pragma HLS array_partition variable=label_list complete dim=0
  // voting boxes
  int vote_list[10];
  #pragma HLS array_partition variable=vote_list complete dim=0

  int pos = 1000;

  // initialize
  INIT_1: for (int i = 0;i < K_CONST; i ++ )
  {
    #pragma HLS unroll
    min_distance_list[i] = 256;
    label_list[i] = 9;
  }

  INIT_2: for (int i = 0;i < 10; i ++ )
  {
    #pragma HLS unroll
    vote_list[i] = 0;
  }

  // go through all the lanes
  // do an insertion sort to keep a sorted neighbor list
  LANES: for (int i = 0; i < PIPELINE_CNT; i ++ )
  {
    INSERTION_SORT_OUTER: for (int j = 0; j < K_CONST; j ++ )
    {
      #pragma HLS pipeline
      pos = 1000;
      INSERTION_SORT_INNER: for (int r = 0; r < K_CONST; r ++ )
      {
        #pragma HLS unroll
        pos = ((knn_set[i][j] < min_distance_list[r]) && (pos > K_CONST)) ? r : pos;
      }

      INSERT: for (int r = K_CONST ;r > 0; r -- )
      {
        #pragma HLS unroll
        if(r-1 > pos)
        {
          min_distance_list[r-1] = min_distance_list[r-2];
          label_list[r-1] = label_list[r-2];
        }
        else if (r-1 == pos)
        {
          min_distance_list[r-1] = knn_set[i][j];
          label_list[r-1] = i / (PIPELINE_CNT / 10);
        }
      }
    }
  }

  // vote
  INCREMENT: for (int i = 0;i < K_CONST; i ++ )
  {
    #pragma HLS pipeline
    vote_list[label_list[i]] += 1;
  }

  LabelType max_vote;
  max_vote = 0;

  // find the maximum value
  VOTE: for (int i = 0;i < 10; i ++ )
  {
    #pragma HLS unroll
    if(vote_list[i] >= vote_list[max_vote])
    {
      max_vote = i;
    }
  }

  result = max_vote;

}

void knn_vote_bitonic_sort( int knn_set[PIPELINE_CNT][K_CONST], LabelType & result )
{
  SortData data[SORT_INPUT_CNT];
  SortId ids[SORT_INPUT_CNT];
  #pragma HLS array_partition variable=data complete dim=1
  #pragma HLS array_partition variable=ids complete dim=1

  for (int i = 0; i < PIPELINE_CNT; i++)
  {
    #pragma HLS unroll
    for (int j = 0; j < K_CONST; j++)
    {
      #pragma HLS unroll
      data[i * K_CONST + j] = knn_set[i][j];
      ids[i * K_CONST + j] = i / (PIPELINE_CNT / 10);
    }
  }

  for (int i = PIPELINE_CNT * K_CONST; i < SORT_INPUT_CNT; i++)
  {
    #pragma HLS unroll
    data[i] = 256;
    ids[i] = 0;
  }

  sorting_network(data, ids);

  unsigned cnts[10];
  for (int i = 0; i < 10; i++)
  {
    #pragma HLS unroll
    cnts[i] = 0;
  }

  for (int i = 0; i < K_CONST; i++)
  {
    #pragma HLS unroll
    cnts[ids[i]]++;
  }

  unsigned max_cnt = cnts[0];
  result = 0;
  for (int i = 1; i < 10; i++)
  {
    #pragma HLS unroll
    if (cnts[i] > max_cnt)
    {
      result = i;
      max_cnt = cnts[i];
    }
  }
}

// Compare the image against a portion of the training images.
void compare_img(WholeDigitType training_set [NUM_TRAINING / PIPELINE_CNT],
                 WholeDigitType test_instance,
                 int knn_set[K_CONST])
{
  TRAINING_LOOP : for ( int i = 0; i < NUM_TRAINING / PIPELINE_CNT; ++i )
  {
    #pragma HLS pipeline

    // Read a new instance from the training set
    WholeDigitType training_instance = training_set[i];

    // Update the KNN set
    update_knn( i == 0, test_instance, training_instance, knn_set );
  }
}

void compare_img_all(WholeDigitType training_set [PIPELINE_CNT][NUM_TRAINING / PIPELINE_CNT],
                     WholeDigitType test_instance,
                     int knn_set[PIPELINE_CNT][K_CONST])
{
  // Compare the image against all training images.
  LANES : for ( int j = 0; j < PIPELINE_CNT; j++ )
  {
    #pragma HLS unroll
    compare_img(training_set[j], test_instance, knn_set[j]);
  }
}

// Recognize one test image.
void recognize_img(WholeDigitType training_set [PIPELINE_CNT][NUM_TRAINING / PIPELINE_CNT],
                   WholeDigitType test_instance,
                   LabelType & result)
{
  #pragma HLS dataflow

  // This array stores K minimum distances per training set
  int knn_set[PIPELINE_CNT][K_CONST];
  #pragma HLS array_partition variable=knn_set complete dim=0

  // Compare the image against all training images.
  compare_img_all(training_set, test_instance, knn_set);

  // Compute the final output
#if BITONIC_SORT == 0
  knn_vote_insert_sort(knn_set, result);
#else
  knn_vote_bitonic_sort(knn_set, result);
#endif
}

// top-level hardware function
// since AXIDMA_SIMPLE interface does not support arrays with size more than 16384 on interface
// we call this function twice to transfer data
void DigitRec(WholeDigitType global_training_set[NUM_TRAINING / 2], WholeDigitType global_test_set[NUM_TEST], LabelType global_results[NUM_TEST], int run) 
{
  static WholeDigitType training_set [PIPELINE_CNT][NUM_TRAINING / PIPELINE_CNT];
  #pragma HLS array_partition variable=training_set complete dim=1

  static WholeDigitType test_set     [NUM_TEST];
  static LabelType results           [NUM_TEST];

  // the first time, just do data transfer and return
  if (run == 0)
  {
    // copy the training set for the first time
	for (int i = 0; i < PIPELINE_CNT / 2; i++ )
      for (int j = 0; j < NUM_TRAINING / PIPELINE_CNT; j++ )
        #pragma HLS pipeline
        training_set[i][j] = global_training_set[i * (NUM_TRAINING / PIPELINE_CNT) + j];
    return;
  }

  // for the second time
  for (int i = 0; i < PIPELINE_CNT / 2; i++ )
    for (int j = 0; j < NUM_TRAINING / PIPELINE_CNT; j++ )
      #pragma HLS pipeline
      training_set[i + PIPELINE_CNT / 2][j] = global_training_set[i * (NUM_TRAINING / PIPELINE_CNT) + j];
  // copy the test set
  for (int i = 0; i < NUM_TEST; i ++ )
    #pragma HLS pipeline
    test_set[i] = global_test_set[i];

  // loop through test set
  TEST_LOOP: for (int t = 0; t < NUM_TEST; ++t) 
  {
    recognize_img(training_set, test_set[t], results[t]);
  }

  // copy the results out
  for (int i = 0; i < NUM_TEST; i ++ )
    #pragma HLS pipeline
    global_results[i] = results[i];

}

