# =============================================== Tools Used in Rosetta =========================================================== #

# sdaccel tools
OCL_CXX   = xcpp
XOCC      = xocc

# sdsoc tools
SDSXX = source $(SDSOC_ROOT)/settings64.sh && sds++

# default sw compiler
SW_CXX = g++

# ============================================= SDAccel Platform and Target Settings ============================================== #

# Set Default OpenCL device and platform
USR_PLATFORM = n
OCL_DEVICE = xilinx:adm-pcie-7v3:1ddr:3.0
OCL_PLATFORM = one_of_default_platforms

# Check if the user specified opencl platform
ifneq ($(OCL_PLATFORM), one_of_default_platforms)
  USR_PLATFORM=y
endif

# Check OCL_TARGET value
OCL_TARGET  = sw_emu
ifeq ($(OCL_TARGET),sw_emu)
else ifeq ($(OCL_TARGET),hw_emu)
else ifeq ($(OCL_TARGET),hw)
else
  $(error "OCL_TARGET does not support the $(OCL_TARGET) value. Supported values are: sw_emu, hw_emu, hw")
endif

# Check opencl kernel file type
OCL_KERNEL_TYPE = ocl

ifeq ($(suffix $(OCL_KERNEL_SRC)),.cl)
  OCL_KERNEL_TYPE=ocl
else
  OCL_KERNEL_TYPE=c
endif

# OpenCL runtime Libraries
OPENCL_INC = $(SDSOC_ROOT)/runtime/include/1_2
OPENCL_LIB = $(SDSOC_ROOT)/runtime/lib/x86_64

# opencl harness files
OCL_HARNESS_DIR     = ../harness/ocl_src
OCL_HARNESS_SRC_CPP = $(OCL_HARNESS_DIR)/CLKernel.cpp $(OCL_HARNESS_DIR)/CLMemObj.cpp $(OCL_HARNESS_DIR)/CLWorld.cpp
OCL_HARNESS_SRC_H   = $(OCL_HARNESS_DIR)/CLKernel.h   $(OCL_HARNESS_DIR)/CLMemObj.h   $(OCL_HARNESS_DIR)/CLWorld.h

# host compilation flags
OCL_HOST_FLAGS = -DOCL -g -lxilinxopencl -I$(OPENCL_INC) $(HOST_INC) -L$(OPENCL_LIB) $(HOST_LIB) -I$(OCL_HARNESS_DIR)

# xclbin compilation flags
XCLBIN_FLAGS = -s -t $(OCL_TARGET) -g -DOCL

ifneq ($(KERNEL_TYPE),ocl)
  XCLBIN_FLAGS += --kernel $(KERNEL_NAME)
endif

ifeq ($(USR_PLATFORM),n)
  XCLBIN_FLAGS += --xdevice $(OCL_DEVICE)
else
  XCLBIN_FLAGS += --platform $(OCL_PLATFORM)
endif

XCLBIN_FLAGS += $(OCL_KERNEL_ARGS)

# host exe
OCL_HOST_EXE        = $(KERNEL_NAME)_host.exe

# Kernel XCLBIN file
XCLBIN        = $(KERNEL_NAME).$(OCL_TARGET).xclbin

# =============================================== SDSoC Platform and Target Settings ============================================== #

DATA_MOVER_SHARING ?= 0
DATA_MOVER_CLOCK ?= 2
KERNEL_CLOCK ?= 2
CLOCK_UNCERTAINTY ?= 30

JOBS ?= 1
THREADS ?= 1

PLATFORM ?= zcu102

TEST_DIR := ..
HUDSON_ROOT ?= $(TEST_DIR)/../../..
SCRIPT_DIR := $(HUDSON_ROOT)/scripts

# platform
SDSOC_PLATFORM := $(PLATFORM_DIR)

# executable
SDSOC_EXE = $(KERNEL_NAME).elf

ifneq ($(words $(SDSOC_KERNEL_SRC)),1)
	SDSOC_KERNEL_EXTRA_SRC := $(wordlist 2, $(words $(SDSOC_KERNEL_SRC)), $(SDSOC_KERNEL_SRC))
	FILES_OPT := -files $(shell echo "$(SDSOC_KERNEL_EXTRA_SRC)" | tr ' ' ',')
endif

# sds++ flags
SDSFLAGS = -sds-pf $(SDSOC_PLATFORM) \
           -sds-hw $(KERNEL_NAME) $(firstword $(SDSOC_KERNEL_SRC)) \
	   $(FILES_OPT) \
	   -clkid $(KERNEL_CLOCK) \
	   -hls-tcl config.tcl \
	   -sds-end \
	   -dm-sharing $(DATA_MOVER_SHARING) \
	   -dmclkid $(DATA_MOVER_CLOCK) \
	   -maxjobs $(JOBS) \
	   -maxthreads $(THREADS)
SDSCFLAGS += -DSDSOC -Wall -O3 -g
SDSCFLAGS += -MMD -MP -MF"$(@:%.o=%.d)"
SDSCFLAGS += $(INC)
SDSCFLAGS += $(DEFINES) -DAXI_BUS_WIDTH=$(AXI_BUS_WIDTH)
ifeq ($(TUNE_INTERF_PARAMS), True)
	SDSCFLAGS += -DTUNE_INTERF_PARAMS
endif
SDSLFLAGS = -O3 

ifeq ($(TRACE), True)
	SDSFLAGS += -trace -trace-buffer 4096
endif

ifdef NO_BITSTREAM
	SDSLFLAGS += -mno-bitstream
endif

# objects
ALL_SDSOC_SRC = $(HOST_SRC_CPP) $(SDSOC_KERNEL_SRC)
OBJECTS := $(ALL_SDSOC_SRC:.cpp=.o)
DEPS := $(OBJECTS:.o=.d)

# ======================================== Software Compilation of Accelerator Code Settings ====================================== #

MIXED_FLAGS = $(SDSCFLAGS) -O0 -I $(subst SDx,Vivado, $(SDSOC_ROOT))/include

MIXED_EXE = $(KERNEL_NAME)_mixed.exe

# =============================================== Pure Software Compilation Settings ============================================== #

# compiler flags
SW_FLAGS = -DSW -O0 -g $(DEFINES) $(INC)

# sw executable
SW_EXE = $(KERNEL_NAME)_sw.exe

# ========================================================= Rules ================================================================= #

# we will have 4 top-level rules: ocl, sdsoc, sw and clean
# default to sw

.PHONY: all ocl sdsoc sw clean

all: sw

# ocl rules
ocl: $(OCL_HOST_EXE) $(XCLBIN)

# ocl secondary rule: host executable
$(OCL_HOST_EXE): $(HOST_SRC_CPP) $(HOST_SRC_H) $(OCL_HARNESS_SRC_CPP) $(OCL_HARNESS_SRC_H) $(DATA)
	$(OCL_CXX) $(OCL_HOST_FLAGS) -o $@ $(HOST_SRC_CPP) $(OCL_HARNESS_SRC_CPP) 

# ocl secondary rule: xclbin 
$(XCLBIN): $(OCL_KERNEL_SRC) $(OCL_KERNEL_H)
	$(XOCC) $(XCLBIN_FLAGS) -o $@ $(OCL_KERNEL_SRC)

# sdsoc rules
sdsoc: $(SDSOC_EXE)

$(SDSOC_EXE): $(OBJECTS)
	$(SDSXX) $(SDSFLAGS) $(SDSLFLAGS) ${OBJECTS} -o $@

-include $(DEPS)

%.o: %.cpp config.tcl $(SDSOC_KERNEL_EXTRA_SRC)
	$(SDSXX) $(SDSFLAGS) -c $(SDSCFLAGS) $< -o $@


# software rules
sw: $(HOST_SRC_CPP) $(HOST_SRC_H) $(SW_KERNEL_SRC) $(SW_KERNEL_H) $(DATA)
	$(SW_CXX) $(SW_FLAGS) -o $(SW_EXE) $(HOST_SRC_CPP) $(SW_KERNEL_SRC)

# mixed build rules
mixed:  $(ALL_SDSOC_SRC)
	$(SW_CXX) $(MIXED_FLAGS) -o $(MIXED_EXE) $(ALL_SDSOC_SRC)

config.tcl:
	@echo "set_clock_uncertainty $(CLOCK_UNCERTAINTY)%" > $@

# cleanup
clean:
	rm -rf *.exe
	rm -rf *.elf
	rm -rf *.xclbin
	rm -rf *.bit
	rm -rf *.rpt
	rm -rf system_estimate.xtxt
	rm -rf _xocc*
	rm -rf _sds
	rm -rf sd_card
	rm -rf .Xil
	rm -rf config.tcl
	rm -rf ./host/*.d
	rm -rf ./sdsoc/*.o
	rm -rf ./sdsoc/*.d
	rm -rf ./host/*.o
