g++ tk.cc build/utils/*.o -O2 -std=c++11 -I. -Iinclude -Lbuild -larm_compute -larm_compute_graph -larm_compute_core -lOpenCL -o tk_test -DARM_COMPUTE_CL
#g++ box.c tk.cc build/utils/*.o -O2 -std=c++11 -I. -Iinclude -Lbuild -larm_compute -larm_compute_graph -larm_compute_core -lOpenCL -o tk_test -DARM_COMPUTE_CL
