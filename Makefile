
all: 
	nvcc --compiler-options='-fPIC'  -use_fast_math -lineinfo --ptxas-options=-v -c XSMD.cu mol_param.cu scat_param.cu env_param.cu WaasKirf.cu XSMD_wrap.cxx 
	nvcc -shared XSMD.o mol_param.o scat_param.o env_param.o WaasKirf.o XSMD_wrap.o -o XSMD.so
XSMD_wrap.cxx: XSMD.i
	swig -c++ -tcl $<
test: speedtest.cu mol_param.cu scat_param.cu env_param.cu WaasKirf.cu coord_ref.cu
	nvcc -use_fast_math -Xptxas=-v,-dlcm=ca -lineinfo   $^
traj: traj_scatter.cu mol_param.cu scat_param.cu env_param.cu WaasKirf.cu
	nvcc -use_fast_math -Xptxas=-v,-dlcm=ca -lineinfo   $^
initial: structure_calc.cu mol_param.cu scat_param.cu env_param.cu WaasKirf.cu coord_ref.cu
	nvcc -use_fast_math -lineinfo --ptxas-options=-v $^ -o structure_calc
fit: fit_initial.cu mol_param.cu scat_param.cu env_param.cu WaasKirf.cu coord_ref.cu
	nvcc -use_fast_math -lineinfo --ptxas-options=-v $^ -o fit_initial
clean:
	rm -rf *.o
	rm XSMD.so
	rm XSMD_wrap.cxx
