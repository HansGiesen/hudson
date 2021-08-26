include ../Makefile.inc

WORK_DIR := ../..
PARAMS_DIR := $(WORK_DIR)/params
LIB_DIR := $(WORK_DIR)/lib/$(PLATFORM)

CFLAGS := -I../utils $(CFLAGS)
LDFLAGS := -L$(LIB_DIR)/minizip -L$(LIB_DIR)/zlib -L../utils -lCraftUtils \
	   $(LDFLAGS)

# OBJ must include a .cpp and .h with same name
OBJ = Accel.o AccelSchedule.o AccelTest.o AccelPrint.o Dense.o InputConv.o
EXE = accel_test_bnn.exe accel_test_layer.exe accel_test_random.exe

GEN_SRC_FILES := Accel.cpp Accel.h AccelSchedule.cpp accel_test_bnn.cpp

all: $(EXE)

# Rule for object files, each object must have a header
$(OBJ): %.o: %.cpp %.h sources
	$(CXX) -c $< -o $@ $(CFLAGS)

%.o: %.cpp sources
	$(CXX) -c $< -o $@ $(CFLAGS)

# Rule for executables
$(EXE): %.exe: %.o $(OBJ)
	$(CXX) $^ -o $@ $(CFLAGS) $(LDFLAGS)

.PHONY: sources hls clean hlsclean

$(GEN_SRC_FILES): sources

sources:
	$(PYTHON3_PATH) ./generate_source.py \
		$(PARAMS_DIR)/cifar10_parameters_nb.zip templates .\
	        $(CONVOLVERS_0) $(CONVOLVERS_1) $(CONVOLVERS_2) \
		$(CONVOLVERS_3) $(CONVOLVERS_4) $(CONVOLVERS_5) \
	        $(CONVOLVERS_6) $(CONVOLVERS_7) $(CONVOLVERS_8) \
		$(WEIGHTS_IN_BRAM) $(PIPELINE) $(BIN_CONV_CNT) \
	        $(BIN_DENSE_CNT) $(WT_MEM_SIZE)

hls:
	vivado_hls hls.tcl

hlsclean:
	rm -rf hls.prj vivado_hls.log

clean: hlsclean
	rm -f *.o *.exe
