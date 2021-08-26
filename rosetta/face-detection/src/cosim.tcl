#=========================================================================
# cosim.tcl
#=========================================================================

set cflags "-g -O0 -I imageLib -DSDSOC -DAXI_BUS_WIDTH=128 -DPIPELINE_CNT=4"
set args "-o output.ppm"

open_project cosim

set_top face_detect

add_files "sdsoc/face_detect.cpp" -cflags $cflags
add_files -tb "host/face_detect_host.cpp host/utils.cpp host/check_result.cpp host/image.cpp" -cflags $cflags

open_solution "solution1" -reset

set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 6.667

config_rtl -reset state

csim_design -argv $args

csynth_design
cosim_design -rtl verilog -trace_level all -argv $args

exit
