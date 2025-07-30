#ifndef EMBEDDING_H
#define EMBEDDING_H

class Embedding
{
public:
    // Constructor adaptado a: (img_height, img_width, patch_size, embed_dim)
    Embedding(int img_height, int img_width, int patch_size, int embed_dim);
    ~Embedding();

    // Forward pass: input [B, N, input_dim] → output [B, N, output_dim]
    void forward(const float *input, float *output, int B, int N);

    // Backward pass: calcula gradientes y ∂L/∂input
    void backward(const float *input, const float *d_output, float *d_input, int B, int N);

    // Actualiza pesos con gradientes y tasa de aprendizaje
    void step(float lr);

    // Resetea los gradientes a cero
    void zero_grad();

    // Guardar y cargar pesos
    void save_weights(const char *filename);
    void load_weights(const char *filename);

private:
    int input_dim;  // Tamaño del vector de entrada (flattened patch)
    int output_dim; // Dimensión del embedding (embed_dim)

    float *W_embed;  // Pesos de embedding [input_dim x output_dim]
    float *dW_embed; // Gradientes de los pesos
};

#endif
