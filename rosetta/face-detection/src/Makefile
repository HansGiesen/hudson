WORK_DIR ?= ..

# Set kernel name
KERNEL_NAME = face_detect

# Set host source and headers
HOST_SRC_CPP = host/face_detect_host.cpp host/utils.cpp host/check_result.cpp host/image.cpp
HOST_SRC_H   = host/utils.h host/check_result.h host/typedefs.h host/image0_320_240.h \
               host/image.h
DATA         = 

# Set host code include paths
HOST_INC = -I$(SDSOC_ROOT)/Vivado_HLS/include
HOST_LIB = -L$(SDSOC_ROOT)/Vivado_HLS/lib

# Set kernel file
OCL_KERNEL_SRC = ocl/face_detect.cpp
OCL_KERNEL_H = host/typedefs.h ocl/*.h
SDSOC_KERNEL_SRC = sdsoc/face_detect.cpp
SDSOC_KERNEL_H = host/typedefs.h sdsoc/*.h
SW_KERNEL_SRC = sw/face_detect_sw.cpp
SW_KERNEL_H = host/typedefs.h sw/*.h

# Set opencl kernel arguments
OCL_KERNEL_ARGS = --max_memory_ports all --report system

DEFINES = -DPIPELINE_CNT=$(PIPELINE_CNT)

#-------------------------
# Leave the rest to harness
#-------------------------
include $(WORK_DIR)/../harness/harness.mk

.PHONY: sources

sources:
	$(PYTHON3_PATH) sdsoc/gen_pipelines.py $(PIPELINE_CNT) > sdsoc/pipelines.h

sdsoc/face_detect.cpp: sources

