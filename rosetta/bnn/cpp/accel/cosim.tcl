#=========================================================================
# hls.tcl
#=========================================================================

set top "top"
set lib_path "../../../../../lib"
set defines "-DCONVOLVERS_0=2 -DCONVOLVERS_1=4 -DCONVOLVERS_2=8 -DCONVOLVERS_3=2 -DCONVOLVERS_4=8 -DCONVOLVERS_5=4 -DCONVOLVERS_6=16 -DCONVOLVERS_7=1 -DCONVOLVERS_8=4 -DWT_MEM_SIZE=37888 -DBATCH_SIZE=1"
set cflags "-DHLS_COMPILE -O3 -I../utils $defines -v"
set ldflags "-L$lib_path/tps/zlib -L$lib_path/tps/minizip -lminizip -laes -lz -v"
set tbflags "-DHLS_COMPILE -O3 -I../utils $defines"
set utils "../utils/Common.cpp ../utils/DataIO.cpp ../utils/ParamIO.cpp ../utils/Timer.cpp ../utils/ZipIO.cpp"
set argv "../../../../.. 1"

open_project cosim

set_top $top

add_files Accel.cpp -cflags $cflags
add_files -tb accel_test_bnn.cpp -cflags $tbflags
add_files -tb AccelSchedule.cpp -cflags $cflags
add_files -tb AccelTest.cpp -cflags $cflags
add_files -tb AccelPrint.cpp -cflags $cflags
add_files -tb InputConv.cpp -cflags $tbflags
add_files -tb Dense.cpp -cflags $tbflags
add_files -tb $utils -cflags $tbflags

open_solution "solution1" -reset

set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 6.667

config_rtl -reset state

csim_design -ldflags $ldflags -argv "$argv" 

csynth_design
cosim_design -rtl verilog -trace_level all -ldflags $ldflags -argv "$argv"

exit
