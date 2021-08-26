/*===============================================================*/
/*                                                               */
/*                        rendering.cpp                          */
/*                                                               */
/*                 C++ kernel for 3D Rendering                   */
/*                                                               */
/*===============================================================*/

#include "../host/typedefs.h"
#include <stdio.h>

//#define DEBUG

/*======================UTILITY FUNCTIONS========================*/


// Determine whether three vertices of a trianlgLe
// (x0,y0) (x1,y1) (x2,y2) are in clockwise order by Pineda algorithm
// if so, return a number > 0
// else if three points are in line, return a number == 0
// else in counterclockwise order, return a number < 0
int check_clockwise( Triangle_2D triangle_2d )
{
  int cw;

  cw = (triangle_2d.x2 - triangle_2d.x0) * (triangle_2d.y1 - triangle_2d.y0)
       - (triangle_2d.y2 - triangle_2d.y0) * (triangle_2d.x1 - triangle_2d.x0);

  return cw;

}

// swap (x0, y0) (x1, y1) of a Triangle_2D
void clockwise_vertices( Triangle_2D *triangle_2d )
{

  bit8 tmp_x, tmp_y;

  tmp_x = triangle_2d->x0;
  tmp_y = triangle_2d->y0;

  triangle_2d->x0 = triangle_2d->x1;
  triangle_2d->y0 = triangle_2d->y1;

  triangle_2d->x1 = tmp_x;
  triangle_2d->y1 = tmp_y;

}


// Given a pixel, determine whether it is inside the triangle
// by Pineda algorithm
// if so, return true
// else, return false
bit1 pixel_in_triangle( bit8 x, bit8 y, Triangle_2D triangle_2d )
{

  int pi0, pi1, pi2;

  pi0 = (x - triangle_2d.x0) * (triangle_2d.y1 - triangle_2d.y0) - (y - triangle_2d.y0) * (triangle_2d.x1 - triangle_2d.x0);
  pi1 = (x - triangle_2d.x1) * (triangle_2d.y2 - triangle_2d.y1) - (y - triangle_2d.y1) * (triangle_2d.x2 - triangle_2d.x1);
  pi2 = (x - triangle_2d.x2) * (triangle_2d.y0 - triangle_2d.y2) - (y - triangle_2d.y2) * (triangle_2d.x0 - triangle_2d.x2);

  return (pi0 >= 0 && pi1 >= 0 && pi2 >= 0);
}

// find the min from 3 integers
bit8 find_min( bit8 in0, bit8 in1, bit8 in2 )
{
  if (in0 < in1)
  {
    if (in0 < in2)
      return in0;
    else 
      return in2;
  }
  else 
  {
    if (in1 < in2) 
      return in1;
    else 
      return in2;
  }
}


// find the max from 3 integers
bit8 find_max( bit8 in0, bit8 in1, bit8 in2 )
{
  if (in0 > in1)
  {
    if (in0 > in2)
      return in0;
    else 
      return in2;
  }
  else 
  {
    if (in1 > in2) 
      return in1;
    else 
      return in2;
  }
}

/*======================PROCESSING STAGES========================*/

// stream input in
void load_input ( axi_bus input[INPUT_WORDS],  Triangle_3D triangle_3ds[NUM_3D_TRI / TASK_PAR][TASK_PAR] )
{
  INPUT_TRI_OUTER: for (int i = 0; i < NUM_3D_TRI / TASK_PAR; i++)
  {
    INPUT_TRI_INNER: for (int j = 0; j < TASK_PAR; j++)
    {
      ap_uint<AXI_BUS_WIDTH * WORDS_PER_TRI> data;
      INPUT_WORDS: for (int k = 0; k < WORDS_PER_TRI; k++)
      {
        #pragma HLS PIPELINE II=1
        unsigned addr = WORDS_PER_TRI * (TASK_PAR * i + j);
        data >>= AXI_BUS_WIDTH;
        data(data.width - 1, data.width - AXI_BUS_WIDTH) = input[addr + k];

        if (k == WORDS_PER_TRI - 1)
        {
          triangle_3ds[i][j].x0 = data( 7,  0);
          triangle_3ds[i][j].y0 = data(15,  8);
          triangle_3ds[i][j].z0 = data(23, 16);
          triangle_3ds[i][j].x1 = data(31, 24);
          triangle_3ds[i][j].y1 = data(39, 32);
          triangle_3ds[i][j].z1 = data(47, 40);
          triangle_3ds[i][j].x2 = data(55, 48);
          triangle_3ds[i][j].y2 = data(63, 56);
          triangle_3ds[i][j].z2 = data(71, 64);
        }
      }
    }
  }
}

// project a 3D triangle to a 2D triangle
void projection ( Triangle_3D triangle_3d, Triangle_2D *triangle_2d, bit2 angle )
{
  #pragma HLS INLINE off
  
  #ifdef DEBUG
    printf("Projection input: %u %u %u %u %u %u %u %u %u\n",
           triangle_3d.x0.to_uint(), triangle_3d.y0.to_uint(),
           triangle_3d.z0.to_uint(), triangle_3d.x1.to_uint(),
           triangle_3d.y1.to_uint(), triangle_3d.z1.to_uint(),
           triangle_3d.x2.to_uint(), triangle_3d.y2.to_uint(),
           triangle_3d.z2.to_uint());
  #endif

  // Setting camera to (0,0,-1), the canvas at z=0 plane
  // The 3D model lies in z>0 space
  // The coordinate on canvas is proportional to the corresponding coordinate 
  // on space
  if(angle == 0)
  {
    triangle_2d->x0 = triangle_3d.x0;
    triangle_2d->y0 = triangle_3d.y0;
    triangle_2d->x1 = triangle_3d.x1;
    triangle_2d->y1 = triangle_3d.y1;
    triangle_2d->x2 = triangle_3d.x2;
    triangle_2d->y2 = triangle_3d.y2;
    triangle_2d->z  = triangle_3d.z0 / 3 + triangle_3d.z1 / 3 + triangle_3d.z2 / 3;
  }

  else if(angle == 1)
  {
    triangle_2d->x0 = triangle_3d.x0;
    triangle_2d->y0 = triangle_3d.z0;
    triangle_2d->x1 = triangle_3d.x1;
    triangle_2d->y1 = triangle_3d.z1;
    triangle_2d->x2 = triangle_3d.x2;
    triangle_2d->y2 = triangle_3d.z2;
    triangle_2d->z  = triangle_3d.y0 / 3 + triangle_3d.y1 / 3 + triangle_3d.y2 / 3;
  }
      
  else if(angle == 2)
  {
    triangle_2d->x0 = triangle_3d.z0;
    triangle_2d->y0 = triangle_3d.y0;
    triangle_2d->x1 = triangle_3d.z1;
    triangle_2d->y1 = triangle_3d.y1;
    triangle_2d->x2 = triangle_3d.z2;
    triangle_2d->y2 = triangle_3d.y2;
    triangle_2d->z  = triangle_3d.x0 / 3 + triangle_3d.x1 / 3 + triangle_3d.x2 / 3;
  }

  #ifdef DEBUG
    printf("Projection output: %u %u %u %u %u %u %u\n",
           triangle_2d->x0.to_uint(), triangle_2d->y0.to_uint(),
           triangle_2d->x1.to_uint(), triangle_2d->y1.to_uint(),
           triangle_2d->x2.to_uint(), triangle_2d->y2.to_uint(),
           triangle_2d->z.to_uint());
  #endif
}

// calculate bounding box for a 2D triangle
bit2 rasterization1 ( Triangle_2D triangle_2d, bit8 max_min[], Triangle_2D *triangle_2d_same)
{
  #pragma HLS INLINE off
  // clockwise the vertices of input 2d triangle
  if ( check_clockwise( triangle_2d ) == 0 )
    return 1;
  if ( check_clockwise( triangle_2d ) < 0 )
    clockwise_vertices( &triangle_2d );

  // copy the same 2D triangle
  triangle_2d_same->x0 = triangle_2d.x0;
  triangle_2d_same->y0 = triangle_2d.y0;
  triangle_2d_same->x1 = triangle_2d.x1;
  triangle_2d_same->y1 = triangle_2d.y1;
  triangle_2d_same->x2 = triangle_2d.x2;
  triangle_2d_same->y2 = triangle_2d.y2;
  triangle_2d_same->z  = triangle_2d.z ;

  // find the rectangle bounds of 2D triangles
  max_min[0] = find_min( triangle_2d.x0, triangle_2d.x1, triangle_2d.x2 );
  max_min[1] = find_max( triangle_2d.x0, triangle_2d.x1, triangle_2d.x2 );
  max_min[2] = find_min( triangle_2d.y0, triangle_2d.y1, triangle_2d.y2 );
  max_min[3] = find_max( triangle_2d.y0, triangle_2d.y1, triangle_2d.y2 );

  #ifdef DEBUG
    unsigned width = max_min[1] - max_min[0] + 1;
    unsigned size = (max_min[1] - max_min[0] + 1) * (max_min[3] - max_min[2] + 1);
    printf("Rasterization 1 output: %u %u %u %u %u %u %u, %u %u %u %u %u, %u\n",
           triangle_2d_same->x0.to_uint(), triangle_2d_same->y0.to_uint(),
           triangle_2d_same->x1.to_uint(), triangle_2d_same->y1.to_uint(),
           triangle_2d_same->x2.to_uint(), triangle_2d_same->y2.to_uint(),
           triangle_2d_same->z.to_uint(),
           max_min[0].to_uint(), max_min[1].to_uint(), max_min[2].to_uint(),
           max_min[3].to_uint(), width, size);
  #endif

  return 0;
}

// find pixels in the triangles from the bounding box
void rasterization2 ( bool skip, bool clear, bit8 max_min[],
                      Triangle_2D triangle_2d,
                      bit8 z_buffer[MAX_X / DATA_PAR][DATA_PAR][MAX_Y],
                      bit8 frame_buffer[MAX_X / DATA_PAR][DATA_PAR][MAX_Y] )
{
  #pragma HLS INLINE off

  if ( clear )
  {
    CLEAR_Y: for ( bit16 i = 0; i < MAX_Y; i++ )
    {
      CLEAR_X: for ( bit16 j = 0; j < MAX_X / DATA_PAR; j++ )
      {
        #pragma HLS PIPELINE II=1
        CLEAR_BANK: for ( bit16 k = 0; k < DATA_PAR; k++)
        {
          z_buffer[j][k][i] = 255;
          frame_buffer[j][k][i] = 0;
        }
      }
    }
  }

  if ( skip )
  {
    return;
  }

  unsigned i = 0;
  unsigned start = max_min[0] / DATA_PAR;
  unsigned end = max_min[1] / DATA_PAR;
  RAST2_Y: for ( unsigned y = max_min[2]; y <= max_min[3]; y++ )
  {
    #pragma HLS loop_tripcount max=8
    RAST2_X_OUTER: for ( unsigned xo = start; xo <= end; xo++ )
    {
      #pragma HLS loop_tripcount max=10
      #pragma HLS PIPELINE II=1
      RAST2_X_INNER: for ( unsigned xi = 0; xi < DATA_PAR; xi++ )
      {
        #pragma HLS DEPENDENCE variable=z_buffer inter false
        #pragma HLS DEPENDENCE variable=frame_buffer inter false
        unsigned x = DATA_PAR * xo + xi;

        if ( pixel_in_triangle( x, y, triangle_2d ) &&
             triangle_2d.z < z_buffer[xo][xi][y] )
        {
          z_buffer[xo][xi][y] = triangle_2d.z;
          frame_buffer[xo][xi][y] = 100;
          #ifdef DEBUG
            printf("Rasterization 2 output %u: %u %u %u %u\n", i, x, y,
                   triangle_2d.z.to_uint(), 100);
            i++;
          #endif
        }
      }
    }               
  }
}

// stream out the frame buffer
void output_FB(bit8 z_buffer[TASK_PAR][MAX_X / DATA_PAR][DATA_PAR][MAX_Y],
               bit8 frame_buffer[TASK_PAR][MAX_X / DATA_PAR][DATA_PAR][MAX_Y],
               axi_bus output[OUTPUT_WORDS])
{
  #pragma HLS INLINE
  OUTPUT_FB_ROW: for ( bit16 i = 0; i < MAX_Y; i++)
  {
    OUTPUT_FB_COL: for ( bit16 j = 0; j < MAX_X; j += AXI_BUS_WIDTH / 8)
    {
      #pragma HLS PIPELINE II=1
      axi_bus out_word;
      OUTPUT_FB_BYTE: for ( bit8 k = 0; k < AXI_BUS_WIDTH / 8; k++)
      {
        unsigned min_z;
        bit8 out_byte;
        OUTPUT_FB_BANK: for ( bit8 l = 0; l < TASK_PAR; l++)
        {
          unsigned byte_addr = (j + k) / DATA_PAR;
          unsigned bank_addr = (j + k) % DATA_PAR;
          bit8 z = z_buffer[l][byte_addr][bank_addr][i];
          if ( l == 0 || z < min_z )
          {
            out_byte = frame_buffer[l][byte_addr][bank_addr][i];
            min_z = z;
          }
        }
        out_word( 8 * k + 7, 8 * k ) = out_byte;
        #ifdef DEBUG
          printf("Output %u %u: %i\n", i.to_uint(), (j + k).to_uint(), out_byte.to_uint());
        #endif
      }
      output[(i * MAX_X + j) / (AXI_BUS_WIDTH / 8)] = out_word;
    }
  }
}

// three stages for processing each 3D triangle
void pipeline( bool clear,
               Triangle_3D triangle_3d,
               bit8 z_buffer[MAX_X / DATA_PAR][DATA_PAR][MAX_Y],
               bit8 frame_buffer[MAX_X / DATA_PAR][DATA_PAR][MAX_Y] )
{
  bit2 angle = 0;
  Triangle_2D triangle_2d;
  Triangle_2D triangle_2d_copy;
  bit8 max_min[4];

  #if USE_DATAFLOW == 1
    #pragma HLS dataflow
  #endif

  // five stages for processing each 3D triangle
  projection( triangle_3d, &triangle_2d, angle );
  bit2 flag = rasterization1( triangle_2d, max_min, &triangle_2d_copy );
  rasterization2( flag, clear, max_min, triangle_2d_copy, z_buffer, frame_buffer );
}


/*========================TOP FUNCTION===========================*/
void rendering( axi_bus input[INPUT_WORDS], axi_bus output[OUTPUT_WORDS])
{
  // local variables
  Triangle_3D triangle_3ds[NUM_3D_TRI / TASK_PAR][TASK_PAR];
#pragma HLS array_partition variable=triangle_3ds complete dim=2

  bit8 z_buffer[TASK_PAR][MAX_X / DATA_PAR][DATA_PAR][MAX_Y];
#pragma HLS array_partition variable=z_buffer complete dim=1
#pragma HLS array_partition variable=z_buffer complete dim=3
  bit8 frame_buffer[TASK_PAR][MAX_X / DATA_PAR][DATA_PAR][MAX_Y];
#pragma HLS array_partition variable=frame_buffer complete dim=1
#pragma HLS array_partition variable=frame_buffer complete dim=3

  load_input( input, triangle_3ds );

  // processing NUM_3D_TRI 3D triangles
  TRIANGLES: for (bit16 i = 0; i < NUM_3D_TRI / TASK_PAR; i++)
  {
    PIPELINES: for (bit8 j = 0; j < TASK_PAR; j++)
    {
      #pragma HLS UNROLL
      pipeline( i == 0, triangle_3ds[i][j], z_buffer[j], frame_buffer[j] );
    }
  }

  // output values: frame buffer
  output_FB(z_buffer, frame_buffer, output);
}
