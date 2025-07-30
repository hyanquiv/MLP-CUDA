#include "transformer.h"
#include "tensor_utils.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cmath>

// Inicialización
Transformer::Transformer(int dim) : dim(dim)
{
    CUDA_CHECK(cudaMalloc(&W_q, dim * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W_k, dim * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W_v, dim * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W_o, dim * dim * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dW_q, dim * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW_k, dim * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW_v, dim * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW_o, dim * dim * sizeof(float)));

    initialize_weights(W_q, dim, dim);
    initialize_weights(W_k, dim, dim);
    initialize_weights(W_v, dim, dim);
    initialize_weights(W_o, dim, dim);
}

// --- Softmax en atención (por bloque)
__global__ void softmax_kernel(float *scores, int B, int N)
{
    int b = blockIdx.x;
    int i = threadIdx.x;
    float *score = scores + b * N * N;

    if (i < N)
    {
        float max_val = -1e9f;
        for (int j = 0; j < N; ++j)
            max_val = fmaxf(max_val, score[i * N + j]);

        float sum = 0.0f;
        for (int j = 0; j < N; ++j)
        {
            score[i * N + j] = expf(score[i * N + j] - max_val);
            sum += score[i * N + j];
        }

        for (int j = 0; j < N; ++j)
            score[i * N + j] /= sum;
    }
}

// --- Softmax backward (derivada aproximada: dA = dA * A * (1 - A))
__global__ void softmax_backward_kernel(float *A, float *dA, int B, int N)
{
    int b = blockIdx.x;
    int i = threadIdx.x;

    if (i < N)
    {
        float *a = A + b * N * N + i * N;
        float *da = dA + b * N * N + i * N;

        for (int j = 0; j < N; ++j)
        {
            float aij = a[j];
            da[j] *= aij * (1.0f - aij);
        }
    }
}

// --- Forward
void transformer_forward(const float *input, float *output,
                         float *W_q, float *W_k, float *W_v, float *W_o,
                         float *Q, float *K, float *V,
                         float *attention_scores,
                         int B, int N, int D)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    // Q = input * W_q
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, N * B, D,
                &alpha, W_q, D, input, D, &beta, Q, D);

    // K = input * W_k
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, N * B, D,
                &alpha, W_k, D, input, D, &beta, K, D);

    // V = input * W_v
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, N * B, D,
                &alpha, W_v, D, input, D, &beta, V, D);

    // Attention scores = Q * K^T
    for (int b = 0; b < B; ++b)
    {
        float *Q_b = Q + b * N * D;
        float *K_b = K + b * N * D;
        float *A_b = attention_scores + b * N * N;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, D,
                    &alpha, Q_b, D, K_b, D, &beta, A_b, N);
    }

    // Softmax atención
    softmax_kernel<<<B, N>>>(attention_scores, B, N);

    // Salida atención = A * V
    float *attn_out;
    CUDA_CHECK(cudaMalloc(&attn_out, B * N * D * sizeof(float)));

    for (int b = 0; b < B; ++b)
    {
        float *A_b = attention_scores + b * N * N;
        float *V_b = V + b * N * D;
        float *O_b = attn_out + b * N * D;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, D, N,
                    &alpha, A_b, N, V_b, D, &beta, O_b, D);
    }

    // output = attn_out * W_o
    for (int b = 0; b < B; ++b)
    {
        float *A_b = attn_out + b * N * D;
        float *O_b = output + b * N * D;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, D, D, &alpha,
                    A_b, D, W_o, D, &beta, O_b, D);
    }

    cudaFree(attn_out);
    cublasDestroy(handle);
}

// --- Backward
void Transformer::backward(
    const float *input, const float *output, const float *d_output,
    float *d_input, int B, int N, int D)
{
    float *Q, *K, *V, *attention_scores;
    CUDA_CHECK(cudaMalloc(&Q, B * N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&K, B * N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&V, B * N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&attention_scores, B * N * N * sizeof(float)));

    // Recomputa forward
    float *attn_out;
    CUDA_CHECK(cudaMalloc(&attn_out, B * N * D * sizeof(float)));
    transformer_forward(input, attn_out,
                        W_q, W_k, W_v, W_o,
                        Q, K, V, attention_scores,
                        B, N, D);

    // dW_o y d_attn_out
    float *d_attn_out;
    CUDA_CHECK(cudaMalloc(&d_attn_out, B * N * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(dW_o, 0, D * D * sizeof(float)));

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    for (int b = 0; b < B; ++b)
    {
        float *d_out_b = (float *)d_output + b * N * D;
        float *attn_b = attn_out + b * N * D;
        float *d_attn_b = d_attn_out + b * N * D;

        // dW_o += attn^T * d_output
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    D, D, N, &alpha,
                    attn_b, D, d_out_b, D,
                    &alpha, dW_o, D);

        // d_attn_out = d_output * W_o^T
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    N, D, D, &alpha,
                    d_out_b, D, W_o, D,
                    &beta, d_attn_b, D);
    }

    // dV = A^T * d_attn, dA = d_attn * V^T
    float *dV, *dA;
    CUDA_CHECK(cudaMalloc(&dV, B * N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dA, B * N * N * sizeof(float)));

    for (int b = 0; b < B; ++b)
    {
        float *A_b = attention_scores + b * N * N;
        float *d_attn_b = d_attn_out + b * N * D;
        float *V_b = V + b * N * D;
        float *dV_b = dV + b * N * D;
        float *dA_b = dA + b * N * N;

        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    D, N, N, &alpha, A_b, N, d_attn_b, D, &beta, dV_b, D);

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    N, N, D, &alpha, d_attn_b, D, V_b, D, &beta, dA_b, N);
    }

    // dA *= softmax grad
    softmax_backward_kernel<<<B, N>>>(attention_scores, dA, B, N);

    // dQ, dK
    float *dQ, *dK;
    CUDA_CHECK(cudaMalloc(&dQ, B * N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, B * N * D * sizeof(float)));

    for (int b = 0; b < B; ++b)
    {
        float *Q_b = Q + b * N * D;
        float *K_b = K + b * N * D;
        float *dA_b = dA + b * N * N;
        float *dQ_b = dQ + b * N * D;
        float *dK_b = dK + b * N * D;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    D, N, N, &alpha, K_b, D, dA_b, N, &beta, dQ_b, D);

        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    D, N, N, &alpha, Q_b, D, dA_b, N, &beta, dK_b, D);
    }

    // dW_q, dW_k, dW_v y d_input
    CUDA_CHECK(cudaMemset(dW_q, 0, D * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(dW_k, 0, D * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(dW_v, 0, D * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_input, 0, B * N * D * sizeof(float)));

    for (int b = 0; b < B; ++b)
    {
        const float *x_b = input + b * N * D;
        float *dQ_b = dQ + b * N * D;
        float *dK_b = dK + b * N * D;
        float *dV_b = dV + b * N * D;
        float *d_inp_b = d_input + b * N * D;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    D, D, N, &alpha, dQ_b, D, x_b, D, &alpha, dW_q, D);

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    D, D, N, &alpha, dK_b, D, x_b, D, &alpha, dW_k, D);

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    D, D, N, &alpha, dV_b, D, x_b, D, &alpha, dW_v, D);

        // d_input = W_q^T dQ + W_k^T dK + W_v^T dV
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    D, N, D, &alpha, W_q, D, dQ_b, D, &beta, d_inp_b, D);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    D, N, D, &alpha, W_k, D, dK_b, D, &alpha, d_inp_b, D);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    D, N, D, &alpha, W_v, D, dV_b, D, &alpha, d_inp_b, D);
    }

    // Liberar
    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dA);
    cudaFree(attn_out);
    cudaFree(d_attn_out);
    cudaFree(attention_scores);
    cublasDestroy(handle);
}
