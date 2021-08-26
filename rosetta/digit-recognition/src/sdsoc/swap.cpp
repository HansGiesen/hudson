// This code is based on work by Yuichi Sugiyama.
//   https://github.com/mmxsrup/bitonic-sort

#include "swap.hpp"


void swap(bool dir, SortData &data1, SortData &data2,
          SortId &id1, SortId &id2) {
  #pragma HLS inline off
  if ((dir && data1 < data2) || (!dir && data1 > data2)) {
    SortData tmp1 = data2;
    data2 = data1;
    data1 = tmp1;
    SortId tmp2 = id2;
    id2 = id1;
    id1 = tmp2;
  }
}
