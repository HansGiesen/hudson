PLATFORM ?= zcu102

include ../../Makefile.inc

WORK_DIR := ../../..
HUDSON_ROOT ?= $(WORK_DIR)/../../..

SCRIPT_DIR := $(HUDSON_ROOT)/scripts
PARAMS_DIR := $(WORK_DIR)/params
LIB_DIR := $(WORK_DIR)/lib/$(PLATFORM)

CFLAGS := -DHLS_COMPILE -I../../utils -I$(LIB_DIR)/zlib $(CFLAGS)
SLDFLAGS := -L$(LIB_DIR)/minizip -L$(LIB_DIR)/zlib -L. -lSdsCraftUtils \
	    $(LDFLAGS)
ifdef NO_BITSTREAM
	SLDFLAGS += -mno-bitstream
endif

SDSFLAGS = -sds-pf $(PLATFORM_DIR) -sds-hw top Accel.cpp -hls-tcl config.tcl \
	   -clkid $(KERNEL_CLOCK) -sds-end -dm-sharing $(DATA_MOVER_SHARING) \
	   -dmclkid $(DATA_MOVER_CLOCK) -maxjobs $(JOBS) -maxthreads $(THREADS)
ifeq (${TRACE}, True)
	SDSFLAGS += -trace -trace-buffer 4096
endif

CXX = source $(SDSOC_ROOT)/settings64.sh && sds++ $(SDSFLAGS)

# OBJ must include a .cpp and .h with same name
UTILS = Common.o Timer.o DataIO.o ParamIO.o ZipIO.o
LIBUTILS = libSdsCraftUtils.a
OBJ = Accel.o AccelSchedule.o AccelTest.o AccelPrint.o Dense.o InputConv.o
EXE = accel_test_bnn.exe

GEN_SRC_FILES := Accel.cpp Accel.h AccelSchedule.cpp accel_test_bnn.cpp

all: $(EXE)

# Rule for object files, each object must have a header
$(OBJ): %.o: ../%.cpp ../%.h sources config.tcl
	$(CXX) -c $< -o $@ $(CFLAGS)

$(UTILS): %.o: ../../utils/%.cpp ../../utils/%.h
	$(CXX) -c $< -o $@ $(CFLAGS)

%.o: ../%.cpp sources
	$(CXX) -c $< -o $@ $(CFLAGS)

# Rule for utils library built by SDSoc
$(LIBUTILS): $(UTILS)
	$(AR) $@ $^

# Rule for executables
$(EXE): %.exe: %.o $(OBJ) $(LIBUTILS)
	$(CXX) $^ -o $@ $(CFLAGS) $(SLDFLAGS)

.PHONY: sources clean

$(addprefix ../, $(GEN_SRC_FILES)): sources

sources:
	$(PYTHON3_PATH) ../generate_source.py \
		$(PARAMS_DIR)/cifar10_parameters_nb.zip ../templates .. \
	        $(CONVOLVERS_0) $(CONVOLVERS_1) $(CONVOLVERS_2) \
		$(CONVOLVERS_3) $(CONVOLVERS_4) $(CONVOLVERS_5) \
	        $(CONVOLVERS_6) $(CONVOLVERS_7) $(CONVOLVERS_8) \
		$(WEIGHTS_IN_BRAM) $(PIPELINE) $(BIN_CONV_CNT) \
	        $(BIN_DENSE_CNT) $(WT_MEM_SIZE)

config.tcl:
	@echo "set_clock_uncertainty $(CLOCK_UNCERTAINTY)%" > $@

clean:
	rm -f *.o *.exe *.bit config.tcl
	rm -rf _sds sd_card
