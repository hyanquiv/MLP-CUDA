#include "embedding.h"
#include "utils.h"
#include <cublas_v2.h>
#include <fstream>
#include <random>

Embedding::Embedding(int input_dim, int output_dim)
    : input_dim(input_dim), output_dim(output_dim)
{
    // Alocar memoria para pesos y gradientes
    CUDA_CHECK(cudaMalloc(&W_embed, input_dim * output_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW_embed, input_dim * output_dim * sizeof(float)));

    // Inicialización aleatoria de pesos
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.02f);
    float *h_weights = new float[input_dim * output_dim];
    for (int i = 0; i < input_dim * output_dim; ++i)
        h_weights[i] = distribution(generator);
    CUDA_CHECK(cudaMemcpy(W_embed, h_weights, input_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_weights;

    zero_grad();
}

Embedding::~Embedding()
{
    cudaFree(W_embed);
    cudaFree(dW_embed);
}

void Embedding::forward(const float *input, float *output, int B, int N)
{
    // input: [B, N, input_dim], output: [B, N, output_dim]
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    for (int b = 0; b < B; ++b)
    {
        const float *in_b = input + b * N * input_dim;
        float *out_b = output + b * N * output_dim;

        // out_b = in_b * W_embed
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    output_dim, N, input_dim,
                    &alpha,
                    W_embed, output_dim,
                    in_b, input_dim,
                    &beta,
                    out_b, output_dim);
    }

    cublasDestroy(handle);
}

void Embedding::backward(const float *input, const float *d_output, float *d_input, int B, int N)
{
    // d_output: [B, N, output_dim], d_input: [B, N, input_dim]
    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int b = 0; b < B; ++b)
    {
        const float *in_b = input + b * N * input_dim;
        const float *d_out_b = d_output + b * N * output_dim;
        float *d_in_b = d_input + b * N * input_dim;

        // dW_embed += in_b^T * d_out_b
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    output_dim, input_dim, N,
                    &alpha,
                    d_out_b, output_dim,
                    in_b, input_dim,
                    &alpha, // acumulativo
                    dW_embed, output_dim);

        // d_input = d_out_b * W_embed^T
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    input_dim, N, output_dim,
                    &alpha,
                    W_embed, output_dim,
                    d_out_b, output_dim,
                    &beta,
                    d_in_b, input_dim);
    }

    cublasDestroy(handle);
}

void Embedding::step(float lr)
{
    int size = input_dim * output_dim;
    update_weights<<<(size + 255) / 256, 256>>>(W_embed, dW_embed, size, lr);
}

void Embedding::zero_grad()
{
    CUDA_CHECK(cudaMemset(dW_embed, 0, input_dim * output_dim * sizeof(float)));
}

void Embedding::save_weights(const char *filename)
{
    float *h_weights = new float[input_dim * output_dim];
    CUDA_CHECK(cudaMemcpy(h_weights, W_embed, input_dim * output_dim * sizeof(float), cudaMemcpyDeviceToHost));
    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<char *>(h_weights), input_dim * output_dim * sizeof(float));
    out.close();
    delete[] h_weights;
}

void Embedding::load_weights(const char *filename)
{
    float *h_weights = new float[input_dim * output_dim];
    std::ifstream in(filename, std::ios::binary);
    in.read(reinterpret_cast<char *>(h_weights), input_dim * output_dim * sizeof(float));
    in.close();
    CUDA_CHECK(cudaMemcpy(W_embed, h_weights, input_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_weights;
}
