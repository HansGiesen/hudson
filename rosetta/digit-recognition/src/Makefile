WORK_DIR ?= ..

# Set kernel name
KERNEL_NAME = DigitRec

# Set host source and headers
HOST_SRC_CPP = host/digit_recognition.cpp host/utils.cpp host/check_result.cpp
HOST_SRC_H   = host/utils.h host/check_result.h host/typedefs.h host/testing_data.h \
               host/training_data.h
DATA         = $(WORK_DIR)/196data/*.dat

# Set host code include paths
HOST_INC = 
HOST_LIB =

# Include paths
INC = -I $(WORK_DIR)/196data

# Set kernel file
OCL_KERNEL_SRC = ocl/digitrec.cl
OCL_KERNEL_H = host/typedefs.h
SDSOC_KERNEL_SRC = sdsoc/digitrec.cpp sdsoc/sorting_network.cpp sdsoc/swap.cpp
SDSOC_KERNEL_H = host/typedefs.h
SW_KERNEL_SRC = sw/digitrec_sw.cpp
SW_KERNEL_H = host/typedefs.h sw/digitrec_sw.h

# Set opencl kernel arguments
OCL_KERNEL_ARGS = --max_memory_ports all --report system

DEFINES = -DPAR_FACTOR=$(PAR_FACTOR)
ifeq ($(BITONIC_SORT), True)
  DEFINES += -DBITONIC_SORT
endif

include $(WORK_DIR)/../harness/harness.mk

.PHONY: sources

sources:
	$(PYTHON3_PATH) sdsoc/gen_sortnet.py sdsoc $(shell expr 30 \* $(PAR_FACTOR)) $(SWAP_CNT)

sdsoc/sorting_network.cpp: sources

