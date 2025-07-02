CC = nvcc
CFLAGS = -std=c++11 -O3 -arch=sm_70
INCLUDES = -Iinclude
SOURCES = src/main.cu src/data_loader.cu src/mlp_cuda.cu src/activation.cu src/cuda_utils.cu
OBJECTS = $(SOURCES:.cu=.o)
EXECUTABLE = build/mnist_mlp

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@

%.o: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)