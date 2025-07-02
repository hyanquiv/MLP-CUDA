#ifndef ACTIVATION_H
#define ACTIVATION_H

__device__ float relu(float x);
__device__ float relu_derivative(float x);
__global__ void softmax(float *input, float *output, int size);
__global__ void relu_forward(float *input, float *output, int size);
__global__ void relu_backward(float *d_output, float *input, float *d_input, int size);

#endif