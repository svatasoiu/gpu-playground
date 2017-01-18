NVCC=nvcc

INCL_DIR=./include
CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64 -lcuda
GCC_OPTS=-O3 -Wall -Wextra -m64 -std=c++11

EXEC=option_pricer

PRICER_DIR=pricers
SEQUENTIAL_PRICER_DIR=$(PRICER_DIR)/cpp
PARALLEL_PRICER_DIR=$(PRICER_DIR)/cuda

SEQUENTIAL_PRICERS=$(wildcard $(SEQUENTIAL_PRICER_DIR)/*.[ct]pp)
PARALLEL_PRICERS=$(wildcard $(PARALLEL_PRICER_DIR)/*.cu)

OBJECTS=$(patsubst %.cpp,%.o,$(SEQUENTIAL_PRICERS))
OBJECTS+=$(patsubst %.cu,%.o,$(PARALLEL_PRICERS))

all: $(EXEC)

$(EXEC): main.o $(OBJECTS)
	$(NVCC) -o $@ $^ -L $(NVCC_OPTS)

main.o: main.cpp
	g++ -c $< $(GCC_OPTS) -I $(INCL_DIR) -I $(CUDA_INCLUDEPATH)

$(SEQUENTIAL_PRICER_DIR)/%.o: $(SEQUENTIAL_PRICER_DIR)/%.cpp
	g++ -c $< $(GCC_OPTS)

$(PARALLEL_PRICER_DIR)/%.o: $(PARALLEL_PRICER_DIR)/%.cu
	nvcc -c $< $(NVCC_OPTS)

run: $(EXEC)
	@./$(EXEC) options.in

clean:
	rm -f $(OBJ_DIR)/*.o $(EXEC)
