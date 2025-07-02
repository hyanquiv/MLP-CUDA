#ifndef MLP_CUDA_H
#define MLP_CUDA_H

#include <vector>
#include "cuda_utils.h"

class MLP
{
public:
    MLP(int input_size, int hidden_size, int output_size);
    MLP(const std::vector<int> &layer_sizes);
    ~MLP();

    void forward(const float *input);
    void backward(const float *input, const int *target, float learning_rate);
    void update_weights(float learning_rate);
    int predict(const float *input);

    const float *get_output() const;
    float get_loss() const;

    struct Impl
    {
        int num_layers;
        std::vector<int> layer_sizes;

        std::vector<float *> activations;
        std::vector<float *> weights;
        std::vector<float *> biases;
        std::vector<float *> d_weights;
        std::vector<float *> d_biases;
        std::vector<float *> deltas;
    };

private:
    Impl *impl;
};

#endif