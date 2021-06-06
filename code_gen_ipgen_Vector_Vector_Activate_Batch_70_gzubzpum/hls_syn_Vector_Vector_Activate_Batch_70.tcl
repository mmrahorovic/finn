
set config_proj_name project_Vector_Vector_Activate_Batch_70
puts "HLS project: $config_proj_name"
set config_hwsrcdir "/workspace/results/code_gen_ipgen_Vector_Vector_Activate_Batch_70_gzubzpum"
puts "HW source dir: $config_hwsrcdir"
set config_proj_part "xcu250-figd2104-2L-e"

set config_bnnlibdir "/workspace/finn-hlslib"

set config_toplevelfxn "Vector_Vector_Activate_Batch_70"
set config_clkperiod 1

open_project $config_proj_name
add_files $config_hwsrcdir/top_Vector_Vector_Activate_Batch_70.cpp -cflags "-std=c++0x -I$config_bnnlibdir"

set_top $config_toplevelfxn
open_solution sol1
set_part $config_proj_part

config_compile -ignore_long_run_time -disable_unroll_code_size_check
config_interface -m_axi_addr64
config_rtl -auto_prefix


create_clock -period $config_clkperiod -name default
csynth_design
export_design -format ip_catalog
exit 0
