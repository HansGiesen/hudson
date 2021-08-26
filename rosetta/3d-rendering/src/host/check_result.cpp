/*===============================================================*/
/*                                                               */
/*                       check_result.cpp                        */
/*                                                               */
/*      Software evaluation of training and test error rate      */
/*                                                               */
/*===============================================================*/

#include <cstdio>
#include "typedefs.h"
#include "output_data.h"

#ifndef SW
bool check_results(axi_bus* output)
#else
bool check_results(bit8 output[MAX_X][MAX_Y])
#endif
{
  #ifndef SW
    bit8 frame_buffer_print[MAX_X][MAX_Y];
  
    // read result from the 32-bit output buffer
    unsigned x = 0;
    unsigned y = 0;
    for (int i = 0; i < OUTPUT_WORDS; i ++ )
    {
      axi_bus word = output[i];
      for (int j = 0; j < AXI_BUS_WIDTH / 8; j ++ )
      {
        frame_buffer_print[x][y] = word(8 * j + 7, 8 * j);
        if (++x == MAX_X)
        {
          x = 0;
          y++;
        }
      }
    }
  #endif

  #if 0
    for (int i = 0; i < MAX_Y; i++)
    {
      printf("{");
      for (int j = 0; j < MAX_X; j++)
      {
        int pix;
        #ifndef SW
          pix = frame_buffer_print[i][j].to_int();
        #else
          pix = output[i][j];
        #endif
        if (j < MAX_X - 1)
          printf("%i, ", pix);
        else
          printf("%i", pix);
      }
      printf("},\n");
    }
  #else
    for (int i = 0; i < MAX_Y; i++)
      for (int j = 0; j < MAX_X; j++)
      {
        int pix;
        #ifndef SW
          pix = frame_buffer_print[i][j].to_int();
        #else
          pix = output[i][j];
        #endif
        if (pix != expected[i][j])
          return true;
      }
  #endif

  return false;
}
