
all: XSMD.o param.o WaasKirf.o XSMD_wrap.o
	nvcc --compiler-options='-fPIC' -maxrregcount 32 -use_fast_math -lineinfo --ptxas-options=-v -c XSMD.cu param.cu WaasKirf.cu XSMD_wrap.cxx 
	nvcc -shared $^ -o XSMD.so
XSMD_wrap.cxx: XSMD.i
	swig -c++ -tcl $<
test: speedtest.cu param.cu WaasKirf.cu coord_ref.cu
	nvcc -maxrregcount 32 -use_fast_math -lineinfo --ptxas-options=-v $^
initial: structure_calc.cu param.cu WaasKirf.cu coord_ref.cu
	nvcc -maxrregcount 32 -use_fast_math -lineinfo --ptxas-options=-v $^ -o structure_calc
clean:
	rm -rf *.o
	rm XSMD.so
