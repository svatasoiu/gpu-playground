NVCC=nvcc

CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include

######################################################
# On Macs the default install locations are below    #
# ####################################################

NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64

GCC_OPTS=-O3 -Wall -Wextra -m64 -std=c++11

EXEC=sort

PARALLEL_SORTS=radix_sort.o parallel_merge_sort.o brick_sort.o hillis_steele_scan.o
SEQUENTIAL_SORTS=sequential_mergesort.o sequential_quicksort.o

all: $(EXEC)

$(EXEC): main.o $(PARALLEL_SORTS) $(SEQUENTIAL_SORTS)
	$(NVCC) -o $@ $^ -L $(NVCC_OPTS)

main.o: main.cpp timer.h utils.h
	g++ -c $< $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

sequential_%.o: sequential_%.cpp
	g++ -c $< $(GCC_OPTS)

%.o: %.cu
	nvcc -c $< $(NVCC_OPTS)

run: $(EXEC)
	./$(EXEC) 16777216

clean:
	rm -f *.o $(EXEC)
