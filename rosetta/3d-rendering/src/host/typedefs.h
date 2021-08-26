/*===============================================================*/
/*                                                               */
/*                          typedefs.h                           */
/*                                                               */
/*                     Typedefs for the host                     */
/*                                                               */
/*===============================================================*/

#ifndef __TYPEDEFS_H__
#define __TYPEDEFS_H__

// resolution 256x256
const int MAX_X = 256;
const int MAX_Y = 256;

// dataset information 
const int NUM_3D_TRI = 3192;

// size of 3D triangle description
const unsigned BITS_PER_TRI = 72;

#ifdef SDSOC
  // number of words needed for one triangle
  const unsigned WORDS_PER_TRI = (BITS_PER_TRI + AXI_BUS_WIDTH - 1) / AXI_BUS_WIDTH;

  // number of input words
  const unsigned INPUT_WORDS = NUM_3D_TRI * WORDS_PER_TRI;
  // number of output words
  const int OUTPUT_WORDS = MAX_X * MAX_Y * 8 / AXI_BUS_WIDTH;
#endif

#ifdef OCL
  #include <string>
  // target device
  // change here to map to a different device
  const std::string TARGET_DEVICE = "xilinx:aws-vu9p-f1:4ddr-xpr-2pr:4.0";
#endif

#ifndef SW
  // hls header
  #include "ap_int.h"
  // specialized datatypes
  typedef ap_uint<1> bit1;
  typedef ap_uint<2> bit2;
  typedef ap_uint<8> bit8;
  typedef ap_uint<16> bit16;
  typedef ap_uint<32> bit32;
#else
  typedef unsigned char bit8;
  typedef unsigned int bit32;
#endif

// struct: 3D triangle
typedef struct
{
  bit8   x0;
  bit8   y0;
  bit8   z0;
  bit8   x1;
  bit8   y1;
  bit8   z1;
  bit8   x2;
  bit8   y2;
  bit8   z2;
} Triangle_3D;

// struct: 2D triangle
typedef struct
{
  bit8   x0;
  bit8   y0;
  bit8   x1;
  bit8   y1;
  bit8   x2;
  bit8   y2;
  bit8   z;
} Triangle_2D;

// struct: candidate pixels
typedef struct
{
  bit8   x;
  bit8   y;
  bit8   z;
  bit8   color;
} CandidatePixel;

// struct: colored pixel
typedef struct
{
  bit8   x;
  bit8   y;
  bit8   color;
} Pixel;

#ifdef SDSOC
  typedef ap_uint<AXI_BUS_WIDTH> axi_bus;
#endif

// dataflow switch
#define ENABLE_DATAFLOW
#endif
