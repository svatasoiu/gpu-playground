NVCC=nvcc

INCL_DIR=./include
CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64 -lcuda
GCC_OPTS=-O3 -Wall -Wextra -m64 -std=c++11 -g

EXEC=option_pricer

BIN_DIR=bin
SRC_DIR=src
TEST_DIR=test

PRICER_DIR=$(SRC_DIR)/pricers

# TODO: make this fully recursive in SRC_DIR

SRCDIRS  := $(SRC_DIR)/pricers $(SRC_DIR)/utils
CPP_SRC  := $(foreach dr, $(SRCDIRS), $(wildcard $(dr)/*.cpp))
CUDA_SRC := $(foreach dr, $(SRCDIRS), $(wildcard $(dr)/*.cu))

# CPP_SRC=$(find $(SRC_DIR) -type f ! -name 'main.cpp' -name '*.cpp') # $(wildcard $(PRICER_DIR)/*.cpp)
# CUDA_SRC=$(find $(SRC_DIR) -type f -name '*.cu') # CUDA_SRC=$(wildcard $(PRICER_DIR)/*.cu)

CPP_OBJECTS=$(patsubst %.cpp,%.o,$(CPP_SRC))
CUDA_OBJECTS=$(patsubst %.cu,%.o,$(CUDA_SRC))

OBJECTS=$(CPP_OBJECTS) $(CUDA_OBJECTS)

all: $(EXEC)

$(EXEC): main.o $(OBJECTS)
	$(NVCC) $^ -o $(BIN_DIR)/$@ -L $(NVCC_OPTS)

main.o: $(SRC_DIR)/main.cpp
	@echo $(CPP_SRC)
	@echo $(CUDA_SRC)
	g++ -c $< $(GCC_OPTS) -I $(INCL_DIR) -I $(CUDA_INCLUDEPATH)

$(CPP_OBJECTS):%.o:%.cpp
	g++ -c $< -o $(basename $@).o $(GCC_OPTS) -I $(INCL_DIR)

$(CUDA_OBJECTS):%.o:%.cu
	nvcc -c $< -o $(basename $@).o $(NVCC_OPTS) -I $(INCL_DIR)

run: $(EXEC)
	@./$(BIN_DIR)/$(EXEC) $(TEST_DIR)/sample_options.in

clean:
	rm -f $(OBJECTS) $(BIN_DIR)/$(EXEC)
