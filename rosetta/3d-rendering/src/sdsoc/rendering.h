/*===============================================================*/
/*                                                               */
/*                         rendering.h                           */
/*                                                               */
/*                 C++ kernel for 3D Rendering                   */
/*                                                               */
/*===============================================================*/

#ifndef __RENDERING_H__
#define __RENDERING_H__

#include "../host/typedefs.h"

#ifdef TUNE_INTERF_PARAMS
#pragma tuner rendering
#else
#pragma SDS data access_pattern(input:SEQUENTIAL, output:SEQUENTIAL)
#endif
void rendering(axi_bus input[INPUT_WORDS], axi_bus output[OUTPUT_WORDS]);

#endif

