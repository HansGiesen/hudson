WORK_DIR ?= ..

# Set kernel name
KERNEL_NAME = optical_flow

# Set host source and headers
HOST_SRC_CPP = host/optical_flow_host.cpp host/utils.cpp host/check_result.cpp imageLib/Convert.cpp \
               imageLib/Convolve.cpp imageLib/flowIO.cpp imageLib/Image.cpp imageLib/ImageIO.cpp imageLib/RefCntMem.cpp
HOST_SRC_H   = host/utils.h host/check_result.h host/typedefs.h imageLib/*.h
DATA         = 

# Set host code include paths
HOST_INC = -I$(SDSOC_ROOT)/Vivado_HLS/include -IimageLib
HOST_LIB = -L$(SDSOC_ROOT)/Vivado_HLS/lib

# Include paths
INC = -I imageLib

# Set kernel file
OCL_KERNEL_SRC = ocl/optical_flow.cpp
OCL_KERNEL_H = host/typedefs.h
SDSOC_KERNEL_SRC = sdsoc/optical_flow.cpp
SDSOC_KERNEL_H = host/typedefs.h
SW_KERNEL_SRC = sw/optical_flow_sw.cpp
SW_KERNEL_H = host/typedefs.h sw/optical_flow_sw.h

# Set opencl kernel arguments
OCL_KERNEL_ARGS = --max_memory_ports all --report system

DEFINES += -DPAR_FACTOR=$(PAR_FACTOR)
DEFINES += -DPIX_WIDTH=$(PIX_WIDTH)
DEFINES += -DOUTER_WIDTH=$(OUTER_WIDTH)
DEFINES += -DTENSOR_WIDTH=$(TENSOR_WIDTH)
DEFINES += -DNOM_WIDTH=$(NOM_WIDTH)
DEFINES += -DDENOM_WIDTH=$(DENOM_WIDTH)
DEFINES += -DGRAD_FILT_WIDTH=$(GRAD_FILT_WIDTH)
DEFINES += -DTENSOR_FILT_WIDTH=$(TENSOR_FILT_WIDTH)
DEFINES += -DVEL_WIDTH=$(VEL_WIDTH)
DEFINES += -DVEL_INT_WIDTH=$(VEL_INT_WIDTH)

#-------------------------
# Leave the rest to harness
#-------------------------
include $(WORK_DIR)/../harness/harness.mk

