#ifndef EMBEDDING_H
#define EMBEDDING_H

class Embedding
{
public:
    float *W_embed;
    float *dW_embed;
    int input_dim;
    int output_dim;

    Embedding(int input_dim, int output_dim);
    ~Embedding();

    void forward(const float *input, float *output, int B, int N);
    void backward(const float *input, const float *d_output, float *d_input, int B, int N);
    void step(float lr);
    void zero_grad();
    void save_weights(const char *filename);
    void load_weights(const char *filename);
};

#endif // EMBEDDING_H
