#include "mlp_cuda.h"
#include "activation.h"
#include "cuda_utils.h"
#include "constants.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <ctime>

// Kernel para inicialización de pesos
__global__ void init_weights_kernel(float *weights, int size, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // Generador de números aleatorios simple
        unsigned int seed = idx * blockIdx.x + threadIdx.x;
        float rnd = static_cast<float>((seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
        rnd = rnd * 2.0f - 1.0f;
        weights[idx] = rnd * scale;
    }
}

// Kernel para operación lineal (Wx + b)
__global__ void linear_forward_kernel(const float *input, const float *weights,
                                      const float *bias, float *output,
                                      int in_size, int out_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_size)
    {
        float sum = 0.0f;
        for (int i = 0; i < in_size; i++)
        {
            sum += input[i] * weights[idx * in_size + i];
        }
        output[idx] = sum + bias[idx];
    }
}

// Kernel para calcular gradientes de pesos
__global__ void weight_gradient_kernel(const float *input, const float *delta,
                                       float *d_weights, int in_size, int out_size)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < out_size && j < in_size)
    {
        atomicAdd(&d_weights[i * in_size + j], input[j] * delta[i]);
    }
}

// Kernel para actualizar pesos
__global__ void update_weights_kernel(float *weights, float *d_weights,
                                      int size, float learning_rate)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        weights[idx] -= learning_rate * d_weights[idx];
        d_weights[idx] = 0.0f; // Resetear gradiente
    }
}

// Kernel para propagación hacia atrás de errores
__global__ void backward_delta_kernel(const float *weights, const float *delta_next,
                                      const float *activation, float *delta,
                                      int in_size, int out_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < in_size)
    {
        float sum = 0.0f;
        for (int i = 0; i < out_size; i++)
        {
            sum += weights[i * in_size + idx] * delta_next[i];
        }
        delta[idx] = sum * relu_derivative(activation[idx]);
    }
}

MLP::MLP(int input_size, int hidden_size, int output_size)
{
    impl = new Impl;
    impl->num_layers = 2; // Capa oculta + capa de salida
    impl->layer_sizes = {input_size, hidden_size, output_size};

    // Verificar que tenemos al menos capa de entrada y salida
    if (impl->num_layers < 1)
    {
        throw std::invalid_argument("Debe haber al menos una capa de entrada y una de salida");
    }

    // Inicializar generador de números aleatorios
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Asignar memoria para activaciones
    impl->activations.resize(impl->num_layers + 1);
    for (int i = 0; i <= impl->num_layers; i++)
    {
        impl->activations[i] = static_cast<float *>(cuda_alloc(impl->layer_sizes[i] * sizeof(float)));
    }

    // Asignar memoria para pesos, bias y gradientes
    impl->weights.resize(impl->num_layers);
    impl->biases.resize(impl->num_layers);
    impl->d_weights.resize(impl->num_layers);
    impl->d_biases.resize(impl->num_layers);
    impl->deltas.resize(impl->num_layers);

    const float weight_scale = 0.1f; // Escala para inicialización de pesos
    const int blockSize = 256;

    for (int i = 0; i < impl->num_layers; i++)
    {
        int in_size = impl->layer_sizes[i];
        int out_size = impl->layer_sizes[i + 1];

        // Capa de pesos
        size_t weight_size = in_size * out_size * sizeof(float);
        impl->weights[i] = static_cast<float *>(cuda_alloc(weight_size));
        impl->d_weights[i] = static_cast<float *>(cuda_alloc(weight_size));

        // Capa de bias
        impl->biases[i] = static_cast<float *>(cuda_alloc(out_size * sizeof(float)));
        impl->d_biases[i] = static_cast<float *>(cuda_alloc(out_size * sizeof(float)));

        // Deltas (solo para capas ocultas)
        if (i < impl->num_layers - 1)
        {
            impl->deltas[i] = static_cast<float *>(cuda_alloc(out_size * sizeof(float)));
        }
        else
        {
            impl->deltas[i] = nullptr; // Capa de salida no necesita delta
        }

        // Inicializar pesos
        int numBlocks = (in_size * out_size + blockSize - 1) / blockSize;
        init_weights_kernel<<<numBlocks, blockSize>>>(impl->weights[i], in_size * out_size, weight_scale);
        cudaDeviceSynchronize();

        // Inicializar bias a cero
        CHECK_CUDA(cudaMemset(impl->biases[i], 0, out_size * sizeof(float)));

        // Inicializar gradientes a cero
        CHECK_CUDA(cudaMemset(impl->d_weights[i], 0, weight_size));
        CHECK_CUDA(cudaMemset(impl->d_biases[i], 0, out_size * sizeof(float)));
    }
}

MLP::MLP(const std::vector<int> &layer_sizes)
{
    impl = new Impl;
    impl->num_layers = layer_sizes.size() - 1;
    impl->layer_sizes = layer_sizes;

    // Verificar que tenemos al menos capa de entrada y salida
    if (impl->num_layers < 1)
    {
        throw std::invalid_argument("Debe haber al menos una capa de entrada y una de salida");
    }

    // Inicializar generador de números aleatorios
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Asignar memoria para activaciones
    impl->activations.resize(impl->num_layers + 1);
    for (int i = 0; i <= impl->num_layers; i++)
    {
        impl->activations[i] = static_cast<float *>(cuda_alloc(layer_sizes[i] * sizeof(float)));
    }

    // Asignar memoria para pesos, bias y gradientes
    impl->weights.resize(impl->num_layers);
    impl->biases.resize(impl->num_layers);
    impl->d_weights.resize(impl->num_layers);
    impl->d_biases.resize(impl->num_layers);
    impl->deltas.resize(impl->num_layers);

    const float weight_scale = 0.1f; // Escala para inicialización de pesos
    const int blockSize = 256;

    for (int i = 0; i < impl->num_layers; i++)
    {
        int in_size = layer_sizes[i];
        int out_size = layer_sizes[i + 1];

        // Capa de pesos
        size_t weight_size = in_size * out_size * sizeof(float);
        impl->weights[i] = static_cast<float *>(cuda_alloc(weight_size));
        impl->d_weights[i] = static_cast<float *>(cuda_alloc(weight_size));

        // Capa de bias
        impl->biases[i] = static_cast<float *>(cuda_alloc(out_size * sizeof(float)));
        impl->d_biases[i] = static_cast<float *>(cuda_alloc(out_size * sizeof(float)));

        // Deltas (solo para capas ocultas)
        if (i < impl->num_layers - 1)
        {
            impl->deltas[i] = static_cast<float *>(cuda_alloc(out_size * sizeof(float)));
        }
        else
        {
            impl->deltas[i] = nullptr; // Capa de salida no necesita delta
        }

        // Inicializar pesos
        int numBlocks = (in_size * out_size + blockSize - 1) / blockSize;
        init_weights_kernel<<<numBlocks, blockSize>>>(impl->weights[i], in_size * out_size, weight_scale);
        cudaDeviceSynchronize();

        // Inicializar bias a cero
        CHECK_CUDA(cudaMemset(impl->biases[i], 0, out_size * sizeof(float)));

        // Inicializar gradientes a cero
        CHECK_CUDA(cudaMemset(impl->d_weights[i], 0, weight_size));
        CHECK_CUDA(cudaMemset(impl->d_biases[i], 0, out_size * sizeof(float)));
    }
}

MLP::~MLP()
{
    // Liberar memoria de activaciones
    for (auto &ptr : impl->activations)
    {
        if (ptr)
            cuda_free(ptr);
    }

    // Liberar pesos y bias
    for (int i = 0; i < impl->num_layers; i++)
    {
        if (impl->weights[i])
            cuda_free(impl->weights[i]);
        if (impl->biases[i])
            cuda_free(impl->biases[i]);
        if (impl->d_weights[i])
            cuda_free(impl->d_weights[i]);
        if (impl->d_biases[i])
            cuda_free(impl->d_biases[i]);
        if (impl->deltas[i])
            cuda_free(impl->deltas[i]);
    }

    delete impl;
}

void MLP::forward(const float *input)
{
    // Copiar entrada al dispositivo
    CHECK_CUDA(cudaMemcpy(impl->activations[0], input,
                          impl->layer_sizes[0] * sizeof(float),
                          cudaMemcpyHostToDevice));

    const int blockSize = 256;

    // Pase hacia adelante a través de todas las capas
    for (int i = 0; i < impl->num_layers; i++)
    {
        int in_size = impl->layer_sizes[i];
        int out_size = impl->layer_sizes[i + 1];

        // Capa lineal: Wx + b
        int numBlocks = (out_size + blockSize - 1) / blockSize;
        linear_forward_kernel<<<numBlocks, blockSize>>>(
            impl->activations[i],     // Entrada
            impl->weights[i],         // Pesos
            impl->biases[i],          // Bias
            impl->activations[i + 1], // Salida (sin activar)
            in_size,
            out_size);
        cudaDeviceSynchronize();

        // Aplicar función de activación (excepto en la capa de salida)
        if (i < impl->num_layers - 1)
        {
            // ReLU para capas ocultas
            relu_forward<<<(out_size + 255) / 256, 256>>>(
                impl->activations[i + 1],
                impl->activations[i + 1],
                out_size);
        }
        else
        {
            // Softmax para capa de salida
            softmax<<<1, out_size>>>(
                impl->activations[i + 1],
                impl->activations[i + 1],
                out_size);
        }
        cudaDeviceSynchronize();
    }
}

void MLP::backward(const float *input, const int *target, float learning_rate)
{
    // Última capa: cálculo del error (diferencia entre predicción y objetivo)
    int output_size = impl->layer_sizes[impl->num_layers];
    float *d_output = static_cast<float *>(cuda_alloc(output_size * sizeof(float)));

    // Copiar salida predicha al host
    float *h_output = new float[output_size];
    CHECK_CUDA(cudaMemcpy(h_output, impl->activations[impl->num_layers],
                          output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Calcular gradiente en el host (para simplificar)
    float *h_d_output = new float[output_size];
    for (int i = 0; i < output_size; i++)
    {
        h_d_output[i] = h_output[i] - (i == *target ? 1.0f : 0.0f);
    }

    // Copiar gradiente al dispositivo
    CHECK_CUDA(cudaMemcpy(d_output, h_d_output,
                          output_size * sizeof(float), cudaMemcpyHostToDevice));

    // Configuración de kernels
    dim3 block(16, 16);

    // Backpropagation a través de las capas
    for (int i = impl->num_layers - 1; i >= 0; i--)
    {
        int in_size = impl->layer_sizes[i];
        int out_size = impl->layer_sizes[i + 1];

        // Calcular gradientes de pesos
        dim3 grid((in_size + 15) / 16, (out_size + 15) / 16);
        weight_gradient_kernel<<<grid, block>>>(
            impl->activations[i], // Activación de entrada
            (i == impl->num_layers - 1) ? d_output : impl->deltas[i],
            impl->d_weights[i],
            in_size,
            out_size);
        cudaDeviceSynchronize();

        // Calcular gradientes de bias
        if (i == impl->num_layers - 1)
        {
            // Capa de salida: gradiente = d_output
            CHECK_CUDA(cudaMemcpy(impl->d_biases[i], d_output,
                                  out_size * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        else
        {
            // Capas ocultas: copiar deltas a d_biases
            CHECK_CUDA(cudaMemcpy(impl->d_biases[i], impl->deltas[i],
                                  out_size * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        // Calcular delta para la capa anterior (excepto para la primera capa)
        if (i > 0)
        {
            int prev_size = impl->layer_sizes[i];
            backward_delta_kernel<<<(prev_size + 255) / 256, 256>>>(
                impl->weights[i], // Pesos de la capa actual
                (i == impl->num_layers - 1) ? d_output : impl->deltas[i],
                impl->activations[i], // Activación de esta capa
                impl->deltas[i - 1],  // Delta para la capa anterior
                in_size,
                out_size);
            cudaDeviceSynchronize();
        }
    }

    // Actualizar pesos
    for (int i = 0; i < impl->num_layers; i++)
    {
        int weight_size = impl->layer_sizes[i] * impl->layer_sizes[i + 1];
        int bias_size = impl->layer_sizes[i + 1];

        int blocks = (weight_size + 255) / 256;
        update_weights_kernel<<<blocks, 256>>>(
            impl->weights[i],
            impl->d_weights[i],
            weight_size,
            learning_rate);
        cudaDeviceSynchronize();

        blocks = (bias_size + 255) / 256;
        update_weights_kernel<<<blocks, 256>>>(
            impl->biases[i],
            impl->d_biases[i],
            bias_size,
            learning_rate);
        cudaDeviceSynchronize();
    }

    // Liberar recursos
    cuda_free(d_output);
    delete[] h_output;
    delete[] h_d_output;
}

void MLP::update_weights(float learning_rate)
{
    // La actualización ya se hace en backward en esta implementación
}

const float *MLP::get_output() const
{
    // Copiar salida al host
    int output_size = impl->layer_sizes[impl->num_layers];
    float *h_output = new float[output_size];
    CHECK_CUDA(cudaMemcpy(h_output, impl->activations[impl->num_layers],
                          output_size * sizeof(float), cudaMemcpyDeviceToHost));
    return h_output;
}

float MLP::get_loss() const
{
    // Implementación simplificada
    return 0.0f;
}

int MLP::predict(const float *input)
{
    forward(input);
    const float *output = get_output();
    int max_idx = 0;
    for (int i = 1; i < impl->layer_sizes[impl->num_layers]; i++)
    {
        if (output[i] > output[max_idx])
        {
            max_idx = i;
        }
    }
    delete[] output;
    return max_idx;
}