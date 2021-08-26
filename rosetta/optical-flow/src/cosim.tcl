#=========================================================================
# cosim.tcl
#=========================================================================

set cflags "-g -O0 -I imageLib -DSDSOC -DAXI_BUS_WIDTH=128 -DPIX_WIDTH=32 -DOUTER_WIDTH=32 -DTENSOR_WIDTH=32 -DNOM_WIDTH=64 -DDENOM_WIDTH=64 -DGRAD_FILT_WIDTH=16 -DTENSOR_FILT_WIDTH=16 -DVEL_WIDTH=32 -DVEL_INT_WIDTH=13"
set args "-p [pwd]/../datasets/current -o output.flo"

open_project cosim

set_top optical_flow

add_files "sdsoc/optical_flow.cpp" -cflags $cflags
add_files -tb "host/optical_flow_host.cpp host/utils.cpp host/check_result.cpp imageLib/Convert.cpp imageLib/Convolve.cpp imageLib/flowIO.cpp imageLib/Image.cpp imageLib/ImageIO.cpp imageLib/RefCntMem.cpp" -cflags $cflags

open_solution "solution1" -reset

set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 6.667

config_rtl -reset state

csim_design -argv $args

csynth_design
cosim_design -rtl verilog -trace_level all -argv $args

exit
