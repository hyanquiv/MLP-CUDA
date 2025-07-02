#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cuda_runtime.h>

// Versión segura para inclusión múltiple
#ifndef RELU_DEFINED
#define RELU_DEFINED
__device__ __forceinline__ float relu(float x)
{
    return x > 0 ? x : 0.0f;
}

__device__ __forceinline__ float relu_derivative(float x)
{
    return x > 0 ? 1.0f : 0.0f;
}
#endif

// Kernels
__global__ void softmax(float *input, float *output, int size);
__global__ void relu_forward(float *input, float *output, int size);
__global__ void relu_backward(float *d_output, float *input, float *d_input, int size);

#endif