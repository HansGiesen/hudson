WORK_DIR ?= ..

# Set kernel name
KERNEL_NAME = SgdLR

# Set host source and headers
HOST_SRC_CPP = ./host/spam_filter.cpp ./host/utils.cpp ./host/check_result.cpp
HOST_SRC_H   = ./host/utils.h ./host/check_result.h ./host/typedefs.h 
DATA         = $(WORK_DIR)/data/*.dat

# Set host code include paths
HOST_INC = -I$(SDSOC_ROOT)/Vivado_HLS/include
HOST_LIB = -L$(SDSOC_ROOT)/Vivado_HLS/lib

# Set kernel file
OCL_KERNEL_SRC = ./ocl/sgd.cpp
OCL_KERNEL_H = ./host/typedefs.h ./ocl/lut.h
SDSOC_KERNEL_SRC = ./sdsoc/sgd.cpp
SDSOC_KERNEL_H = ./host/typedefs.h ./sdsoc/lut.h
SW_KERNEL_SRC = ./sw/sgd_sw.cpp
SW_KERNEL_H = ./host/typedefs.h

# Set opencl kernel arguments
OCL_KERNEL_ARGS = -DOCL --max_memory_ports all --report system

DEFINES = -DPAR_FACTOR=$(PAR_FACTOR)

#-------------------------
# Leave the rest to harness
#-------------------------
include $(WORK_DIR)/../harness/harness.mk

