CC=g++
NVCC=nvcc

GXXFLAGS=
CUDAFLAGS= -c
LIBS = -lcuda -lcudart
CUDALIB=-L/opt/cuda/lib64
CUDAINC=-I/opt/cuda/include
INCDIRS=-I./include

bin=bin/
src=src/
include=include/
build=build/

all: $(build)numc.o
	$(CC) -o $(bin)main $(CUDALIB) $(CUDAINC) $(INCDIRS) $(LIBS) $(GXXFLAGS) $(src)main.cpp $(build)numc.o 

$(build)numc.o: $(src)numc.cu $(include)numc.cuh
	$(NVCC) $(CUDAFLAGS) $(INCDIRS) $(src)numc.cu -o $(build)numc.o
