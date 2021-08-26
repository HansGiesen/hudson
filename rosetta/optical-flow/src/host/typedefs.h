/*===============================================================*/
/*                                                               */
/*                        typedefs.h                             */
/*                                                               */
/*        Defines types and constants for host function          */
/*                                                               */
/*===============================================================*/

#ifndef __TYPEDEFS_H__
#define __TYPEDEFS_H__

#include <gmp.h>
#define __gmp_const const

//#include "ap_fixed.h"
const int MAX_HEIGHT = 436;
const int MAX_WIDTH = 1024;

const int INPUT_LENGTH = MAX_HEIGHT * MAX_WIDTH / PAR_FACTOR;
const int OUTPUT_LENGTH = MAX_HEIGHT * MAX_WIDTH / PAR_FACTOR;

// basic typedefs
#ifdef SDSOC
  #include "ap_fixed.h"
  typedef ap_ufixed<8,0> input_t;
  typedef ap_fixed<PIX_WIDTH,4> pixel_t;
  typedef ap_fixed<OUTER_WIDTH,1> outer_pixel_t;
  typedef ap_fixed<TENSOR_WIDTH,1> tensor_pixel_t;
  typedef ap_fixed<NOM_WIDTH,1> nom_t;
  typedef ap_fixed<DENOM_WIDTH,1> denom_t;
  typedef ap_fixed<VEL_WIDTH,VEL_INT_WIDTH> vel_pixel_t;
  typedef ap_ufixed<GRAD_FILT_WIDTH,0> grad_filt_t;
  typedef ap_ufixed<TENSOR_FILT_WIDTH,0> tensor_filt_t;

  typedef ap_uint<8*PAR_FACTOR> input_vec_t;
  typedef ap_uint<PIX_WIDTH*PAR_FACTOR> pixel_vec_t;
  typedef ap_uint<OUTER_WIDTH*PAR_FACTOR> outer_pixel_vec_t;
  typedef ap_uint<TENSOR_WIDTH*PAR_FACTOR> tensor_pixel_vec_t;
  typedef ap_uint<VEL_WIDTH*PAR_FACTOR> vel_pixel_vec_t;

  typedef ap_uint<AXI_BUS_WIDTH> axi_bus_t;
#endif
#ifdef OCL
  #include "ap_fixed.h"
  typedef ap_fixed<17,9> input_t;
  typedef ap_fixed<32,13> pixel_t;
  typedef ap_fixed<32,27> outer_pixel_t;
  typedef ap_fixed<64,56> nom_t;
  typedef ap_fixed<32,13> vel_pixel_t;
#endif
#ifdef SW
  typedef float pixel_t;
  typedef float outer_pixel_t;
  typedef float tensor_pixel_t;
  typedef float vel_pixel_t;
#endif

typedef struct{
  pixel_t x;
  pixel_t y;
  pixel_t z;
}gradient_t;

typedef struct{
  outer_pixel_t val[6];
}outer_t; 

typedef struct{
  tensor_pixel_t val[6];
}tensor_t;

typedef struct{
  vel_pixel_t x;
  vel_pixel_t y;
}velocity_t;

#ifdef SDSOC
  typedef struct{
    pixel_vec_t x;
    pixel_vec_t y;
    pixel_vec_t z;
  }gradient_vec_t;

  typedef struct{
    outer_pixel_vec_t val[6];
  }outer_vec_t; 

  typedef struct{
    tensor_pixel_vec_t val[6];
  }tensor_vec_t;

  typedef struct{
    vel_pixel_vec_t x;
    vel_pixel_vec_t y;
  }velocity_vec_t;
#endif

#ifndef SW
  #include "ap_int.h"
  // for data packing
  typedef ap_uint<64> frames_t;
#endif

#ifdef OCL
  #include <string>
  // change the target device here
  const std::string TARGET_DEVICE = "xilinx:aws-vu9p-f1:4ddr-xpr-2pr:4.0";
#endif

#ifdef __SYNTHESIS__
  #define STATIC
#else
  #define STATIC static
#endif

#endif
