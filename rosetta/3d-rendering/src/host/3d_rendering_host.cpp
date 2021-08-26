/*===============================================================*/
/*                                                               */
/*                       3d_rendering.cpp                        */
/*                                                               */
/*      Main host function for the 3D Rendering application.     */
/*                                                               */
/*===============================================================*/

// standard C/C++ headers
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <time.h>
#include <sys/time.h>

#ifdef OCL
  // harness headers
  #include "CLWorld.h"
  #include "CLKernel.h"
  #include "CLMemObj.h"
  // harness namespace
  using namespace rosetta;
#endif

#ifdef SDSOC
#ifdef __SDSCC__
  // sdsoc headers
  #include "sds_lib.h"
#endif
  // hardware function declaration
  #include "../sdsoc/rendering.h"
#endif

#ifdef SW
  # include "../sw/rendering_sw.h"
#endif

// other headers
#include "utils.h"
#include "typedefs.h"
#include "check_result.h"
#ifdef SDSOC
  #ifdef TUNE_INTERF_PARAMS
    #include "hudson.h"
  #endif
#endif

// data
#include "input_data.h"


int main(int argc, char ** argv) 
{
  printf("3D Rendering Application\n");

  #ifdef OCL
    // parse command line arguments for opencl version
    std::string kernelFile("");
    parse_sdaccel_command_line_args(argc, argv, kernelFile);
  #endif
  // sdsoc and sw versions have no additional command line arguments

  // for this benchmark, data is included in array triangle_3ds

  // timers
  struct timeval start, end;

  // opencl version host code
  #ifdef OCL

    // create space for input and output
    bit32* input  = new bit32[3 * NUM_3D_TRI];
    bit32* output = new bit32[NUM_FB];
  
    // pack input data for better performance
    for ( int i = 0; i < NUM_3D_TRI; i ++)
    {
      input[3*i](7,0)     = triangle_3ds[i][0];
      input[3*i](15,8)    = triangle_3ds[i][1];
      input[3*i](23,16)   = triangle_3ds[i][2];
      input[3*i](31,24)   = triangle_3ds[i][3];
      input[3*i+1](7,0)   = triangle_3ds[i][4];
      input[3*i+1](15,8)  = triangle_3ds[i][5];
      input[3*i+1](23,16) = triangle_3ds[i][6];
      input[3*i+1](31,24) = triangle_3ds[i][7];
      input[3*i+2](7,0)   = triangle_3ds[i][8];
      input[3*i+2](31,8)  = 0;
    }
  
    // create OpenCL world
    CLWorld rendering_world = CLWorld(TARGET_DEVICE, CL_DEVICE_TYPE_ACCELERATOR);
  
    // add the bitstream file
    rendering_world.addProgram(kernelFile);
  
    // create kernels
    // this kernel is written in C++
    CLKernel Rendering(rendering_world.getContext(), rendering_world.getProgram(), "rendering", rendering_world.getDevice());
  
    // create mem objects
    CLMemObj input_mem ( (void*)input,  sizeof(bit32), 3 * NUM_3D_TRI, CL_MEM_READ_ONLY);
    CLMemObj output_mem( (void*)output, sizeof(bit32), NUM_FB,         CL_MEM_WRITE_ONLY);
  
    // start timer
    gettimeofday(&start, 0);
  
    // add them to the world
    // added in sequence, each of them can be referenced by an index
    rendering_world.addMemObj(input_mem);
    rendering_world.addMemObj(output_mem);
  
    // set work size
    int global_size[3] = {1, 1, 1};
    int local_size[3] = {1, 1, 1};
    Rendering.set_global(global_size);
    Rendering.set_local(local_size);
  
    // add them to the world
    rendering_world.addKernel(Rendering);
  
    // set kernel arguments
    rendering_world.setMemKernelArg(0, 0, 0);
    rendering_world.setMemKernelArg(0, 1, 1);
  
    // run!
    rendering_world.runKernels();
  
    // read the data back
    rendering_world.readMemObj(1);
    
    // end timer
    gettimeofday(&end, 0);
  #endif

  #ifdef SDSOC
    // create space for input and output
    #ifndef TUNE_INTERF_PARAMS
      axi_bus *input = (axi_bus*)sds_alloc(INPUT_WORDS * sizeof(axi_bus));
      axi_bus* output = (axi_bus*)sds_alloc(OUTPUT_WORDS * sizeof(axi_bus));
    #else
      axi_bus *input = (axi_bus*)alloc_input(INPUT_WORDS * sizeof(axi_bus));
      axi_bus* output = (axi_bus*)alloc_output(OUTPUT_WORDS * sizeof(axi_bus));
    #endif

    // pack input data for better performance
    for ( int i = 0; i < NUM_3D_TRI; i++)
    {
      for ( unsigned j = 0; j < WORDS_PER_TRI; j++)
      {
        ap_uint<WORDS_PER_TRI * AXI_BUS_WIDTH> word = 0;
        for ( unsigned k = 0; k < AXI_BUS_WIDTH / 8; k++)
        {
          unsigned lsb = j * AXI_BUS_WIDTH + 8 * k;
          if (lsb < BITS_PER_TRI)
            word(lsb + 7, lsb) = triangle_3ds[i][lsb / 8];
          else
            word(lsb + 7, lsb) = 0;
        }
        input[i * WORDS_PER_TRI + j] = word;
      }
    }

    // run hardware function and time it

    gettimeofday(&start, 0);
    #ifdef __SDSCC__
      unsigned long long start_time = sds_clock_counter();
    #endif
    rendering(input, output);
    #ifdef __SDSCC__
      unsigned long long end_time = sds_clock_counter();
    #endif
    gettimeofday(&end, 0);

  #endif

  #ifdef SW
    //input
    Triangle_3D input[NUM_3D_TRI];
    for ( int i = 0; i < NUM_3D_TRI; i++)
    {
      input[i].x0     = triangle_3ds[i][0];
      input[i].y0     = triangle_3ds[i][1];
      input[i].z0     = triangle_3ds[i][2];
      input[i].x1     = triangle_3ds[i][3];
      input[i].y1     = triangle_3ds[i][4];
      input[i].z1     = triangle_3ds[i][5];
      input[i].x2     = triangle_3ds[i][6];
      input[i].y2     = triangle_3ds[i][7];
      input[i].z2     = triangle_3ds[i][8];
    }

    // output
    bit8 output[MAX_X][MAX_Y];
    // run and time sw function
    gettimeofday(&start, 0);
    #ifdef __SDSCC__
      unsigned long long start_time = sds_clock_counter();
    #endif
    rendering_sw(input, output);
    #ifdef __SDSCC__
      unsigned long long end_time = sds_clock_counter();
    #endif
    gettimeofday(&end, 0);
  #endif 
 
  // check results
  bool error = check_results(output);
 
  // print time
  long long elapsed = (end.tv_sec - start.tv_sec) * 1000000LL + end.tv_usec - start.tv_usec;   
  printf("elapsed time: %lld us\n", elapsed);

  #ifdef __SDSCC__
    unsigned long long duration = end_time - start_time;
    printf("The hardware test took %llu cycles.\n", duration);
  #endif

  int exit_code;
  if (!error){
    printf("TEST PASSED\n");
    exit_code = 0;
  }
  else{
    printf("TEST FAILED\n");
    exit_code = 1;
  }

  // cleanup
  #ifdef OCL
    rendering_world.releaseWorld();
    delete []input;
    delete []output;
  #endif

  #ifdef SDSOC
    #ifndef TUNE_INTERF_PARAMS
      sds_free(input);
      sds_free(output);
    #else
      free_input(input);
      free_output(output);
    #endif
  #endif

  return exit_code;

}
