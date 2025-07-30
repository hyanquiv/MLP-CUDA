#include "tensor_utils.h"
#include <random>
#include <cmath>

void initialize_embedding_weights(float *host_weights, float *host_bias, int patch_dim, int embed_dim)
{
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f / sqrtf(patch_dim));

    for (int i = 0; i < patch_dim * embed_dim; ++i)
        host_weights[i] = dist(gen);

    for (int i = 0; i < embed_dim; ++i)
        host_bias[i] = 0.0f;
}

void initialize_transformer_weights(float *W_q, float *W_k, float *W_v, int embed_dim)
{
    std::mt19937 gen(1337);                                              // Semilla distinta
    std::normal_distribution<float> dist(0.0f, 1.0f / sqrtf(embed_dim)); // Xavier

    int size = embed_dim * embed_dim;

    for (int i = 0; i < size; ++i)
    {
        W_q[i] = dist(gen);
        W_k[i] = dist(gen);
        W_v[i] = dist(gen);
    }
}
