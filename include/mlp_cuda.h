#ifndef MLP_CUDA_H
#define MLP_CUDA_H

class MLP
{
public:
    MLP(int input_size, int hidden_size, int output_size);
    ~MLP();

    void forward(const float *input);
    void backward(const float *input, const int *target, float learning_rate);
    void update_weights(float learning_rate);
    int predict(const float *input);

    // Getters para acceso controlado
    const float *get_output() const;
    float get_loss() const;

private:
    // Estructura interna que oculta implementación CUDA
    struct Impl;
    Impl *impl;
};

#endif