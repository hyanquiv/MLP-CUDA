#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <cuda_runtime.h>

class Transformer
{
public:
    float *W_q, *W_k, *W_v, *W_o;
    float *dW_q, *dW_k, *dW_v, *dW_o;

    Transformer(int D);
    ~Transformer();

    void forward(
        const float *input, // [B, N, D]
        float *output,      // [B, N, D]
        int B, int N, int D);

    void backward(
        const float *input,    // [B, N, D]
        const float *output,   // [B, N, D]
        const float *d_output, // [B, N, D]
        float *d_input,        // [B, N, D]
        int B, int N, int D);

    void step(float lr);
    void zero_grad();
    void save_weights(const char *filename);
    void load_weights(const char *filename);
};

#endif // TRANSFORMER_H
