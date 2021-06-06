#!/bin/sh
lli=${LLVMINTERP-lli}
exec $lli \
    /workspace/results/code_gen_ipgen_Vector_Vector_Activate_Batch_70_gzubzpum/project_Vector_Vector_Activate_Batch_70/sol1/.autopilot/db/a.g.bc ${1+"$@"}
