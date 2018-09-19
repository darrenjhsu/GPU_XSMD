

CC := nvcc
SRCDIR := src
BUILDDIR := build
TARGET := bin/XSMD.so

SRCEXT := cu
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
PARAMS := $(SRCDIR)/mol_param.cu $(SRCDIR)/scat_param.cu $(SRCDIR)/env_param.cu $(SRCDIR)/WaasKirf.cu $(SRCDIR)/coord_ref.cu 
FITPARAMS := $(SRCDIR)/mol_param.cu $(SRCDIR)/env_param.cu $(SRCDIR)/WaasKirf.cu $(SRCDIR)/coord_ref.cu $(SRCDIR)/expt_data.cu
SOURCESOBJ := $(patsubst $(SRCDIR)/%, $(BUILDDIR)/%, $(SOURCES:.$(SRCEXT)=.o))
PARAMSOBJ := $(patsubst $(SRCDIR)/%, $(BUILDDIR)/%, $(PARAMS:.$(SRCEXT)=.o))
FITPARAMSOBJ := $(patsubst $(SRCDIR)/%, $(BUILDDIR)/%, $(FITPARAMS:.$(SRCEXT)=.o))
CFLAGS := --compiler-options='-fPIC' -use_fast_math -lineinfo --ptxas-options=-v
LIB := -lgsl -lgslcblas -lm
INC := -Iinclude
GSLINC := -I/home/djh992/lib/gsl/include
GSLLIB := -L/home/djh992/lib/gsl/lib


$(TARGET): $(BUILDDIR)/XSMD_wrap.o $(PARAMSOBJ)
	$(CC) -shared $^ -o $(TARGET) 
	#nvcc --compiler-options='-fPIC' -use_fast_math -lineinfo --ptxas-options=-v -c XSMD.cu mol_param.cu scat_param.cu env_param.cu WaasKirf.cu XSMD_wrap.cxx 
	#nvcc -shared XSMD.o mol_param.o scat_param.o env_param.o WaasKirf.o XSMD_wrap.o -o XSMD.so
test: speedtest.cu $(PARAMSOBJ)
	nvcc -use_fast_math -Xptxas=-v,-dlcm=ca -lineinfo   $^
traj: $(BUILDDIR)/traj_scatter.o $(PARAMSOBJ)
	$(CC) $(CFLAGS) $^ -o bin/traj_scatter.out
initial: $(BUILDDIR)/structure_calc.o $(PARAMSOBJ)
	$(CC) $(CFLAGS) $^ -o bin/structure_calc.out
fit: $(BUILDDIR)/fit_initial.o $(FITPARAMSOBJ) 
	@echo "Linking for fit ......"
	nvcc $(GSLLIB) $^ $(LIB) -o bin/fit_initial.out 
$(BUILDDIR)/XSMD_wrap.o: $(SRCDIR)/XSMD_wrap.cxx
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) $(GSLINC) $(LIB) -c -o $@ $^
$(SRCDIR)/XSMD_wrap.cxx: $(SRCDIR)/XSMD.i
	swig -c++ -tcl $(INC) $^
$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) $(GSLINC) $(LIB) -c -o $@ $^

clean:
	@echo " Cleaning..."; 
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)	
	@echo " $(RM) -r $(SRCDIR)/XSMD_wrap.cxx"; $(RM) -r $(SRCDIR)/XSMD_wrap.cxx

