WORK_DIR ?= ..

# Set kernel name
KERNEL_NAME = rendering

# Set host source and headers
HOST_SRC_CPP = ./host/3d_rendering_host.cpp ./host/utils.cpp ./host/check_result.cpp
HOST_SRC_H   = ./host/utils.h ./host/check_result.h ./host/typedefs.h ./host/input_data.h 
DATA         = 

# Set host code include paths
HOST_INC = -I$(SDSOC_ROOT)/Vivado_HLS/include
HOST_LIB = -L$(SDSOC_ROOT)/Vivado_HLS/lib

# Set kernel file
OCL_KERNEL_SRC = ./ocl/rendering.cpp
OCL_KERNEL_H = ./host/typedefs.h
SDSOC_KERNEL_SRC = ./sdsoc/rendering.cpp
SDSOC_KERNEL_H = ./host/typedefs.h
SW_KERNEL_SRC = ./sw/rendering_sw.cpp
SW_KERNEL_H = ./host/typedefs.h ./sw/rendering_sw.h

# Set opencl kernel arguments
OCL_KERNEL_ARGS = --max_memory_ports all --report system

DEFINES = -DDATA_PAR=$(DATA_PAR) -DTASK_PAR=$(TASK_PAR)
ifeq ($(USE_DATAFLOW), True)
	DEFINES += -DUSE_DATAFLOW
endif

#-------------------------
# Leave the rest to harness
#-------------------------
include $(WORK_DIR)/../harness/harness.mk

