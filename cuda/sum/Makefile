NVCC=nvcc

###################################
# These are the default install   #
# locations on most linux distros #
###################################

OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include

###################################################
# On Macs the default install locations are below #
###################################################

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui

CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include

######################################################
# On Macs the default install locations are below    #
# ####################################################

NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64

GCC_OPTS=-O3 -Wall -Wextra -m64

EXEC=sum

all: $(EXEC)

sum: main.o parallel_sum.o
	$(NVCC) -o $@ $^ -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

main.o: main.cpp timer.h utils.h
	g++ -c $< $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH)

parallel_sum.o: parallel_sum.cu
	nvcc -c $< $(NVCC_OPTS)

run: $(EXEC)
	./$(EXEC) 33554432

clean:
	rm -f *.o $(EXEC)
