NVCC=nvcc

CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64 -lcuda
GCC_OPTS=-O3 -Wall -Wextra -m64 -std=c++11

EXEC=transpose

PARALLEL_TRANSPOSES=parallel_transpose_v1.o parallel_transpose_v2_v3.o parallel_transpose_v4.o parallel_transpose_v5.o
SEQUENTIAL_TRANPOSES=sequential_transpose_v1.o

all: $(EXEC)

$(EXEC): main.o $(PARALLEL_TRANSPOSES) $(SEQUENTIAL_TRANPOSES)
	$(NVCC) -o $@ $^ -L $(NVCC_OPTS)

main.o: main.cpp timer.h utils.h
	g++ -c $< $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

sequential_%.o: sequential_%.cpp
	g++ -c $< $(GCC_OPTS)

%.o: %.cu
	nvcc -c $< $(NVCC_OPTS)

run: $(EXEC)
	@./$(EXEC) 16 64 71 128 512 1024 2048 4096 

clean:
	rm -f *.o $(EXEC)
