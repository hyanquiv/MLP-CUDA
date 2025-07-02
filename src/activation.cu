#include "activation.h"
#include <cuda_runtime.h>
#include <math.h>

// Implementación de ReLU y su derivada
__device__ float relu(float x)
{
    return x > 0 ? x : 0;
}

__device__ float relu_derivative(float x)
{
    return x > 0 ? 1 : 0;
}

// Kernel para aplicar ReLU forward
__global__ void relu_forward(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = relu(input[idx]);
    }
}

// Kernel para aplicar ReLU backward
__global__ void relu_backward(float *d_output, float *input, float *d_input, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        d_input[idx] = d_output[idx] * relu_derivative(input[idx]);
    }
}

// Kernel para aplicar softmax
__global__ void softmax(float *input, float *output, int size)
{
    // Un solo bloque maneja todo el vector
    if (blockIdx.x == 0)
    {
        // Encontrar el valor máximo para estabilidad numérica
        float max_val = input[0];
        for (int i = 1; i < size; i++)
        {
            if (input[i] > max_val)
            {
                max_val = input[i];
            }
        }

        // Calcular suma exponencial
        float exp_sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            exp_sum += expf(input[i] - max_val);
        }

        // Calcular softmax
        for (int i = 0; i < size; i++)
        {
            output[i] = expf(input[i] - max_val) / exp_sum;
        }
    }
}