#include "activation.h"

__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

__device__ float relu_derivative(float x) {
    return x > 0 ? 1 : 0;
}

__global__ void relu_forward(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        output[idx] = relu(input[idx]);
    }
}

// Implementaciones para otras funciones de activación...