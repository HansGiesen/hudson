#=========================================================================
# cosim.tcl
#=========================================================================

set cflags "-g -O0 -I [pwd]/../196data -DSDSOC -DPAR_FACTOR=4 -DBITONIC_SORT -DSWAP_CNT=16 -DAXI_BUS_WIDTH=64"

open_project cosim

set_top DigitRec

add_files "sdsoc/digitrec.cpp sdsoc/sorting_network.cpp sdsoc/swap.cpp" -cflags $cflags
add_files -tb "host/digit_recognition.cpp host/utils.cpp host/check_result.cpp" -cflags $cflags

open_solution "solution1" -reset

set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 6.667

config_rtl -reset state

csim_design 

csynth_design
cosim_design -rtl verilog -trace_level all

exit
