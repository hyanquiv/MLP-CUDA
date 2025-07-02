CC = nvcc
CFLAGS = -std=c++11 -O3 -arch=sm_75  # CAMBIA sm_75 por tu arquitectura
INCLUDES = -Iinclude
LIBS = -lcublas -lcurand
SOURCES = src/main.cu src/data_loader.cu src/mlp_cuda.cu src/activation.cu src/cuda_utils.cu src/train.cu
OBJECTS = $(SOURCES:.cu=.o)
EXECUTABLE = build/mnist_mlp

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	@mkdir -p build
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LIBS)

%.o: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
	rm -rf build

run: $(EXECUTABLE)
	./$(EXECUTABLE)

.PHONY: all clean run