#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

void initialize_embedding_weights(float *host_weights, float *host_bias, int patch_dim, int embed_dim);
void initialize_transformer_weights(float *W_q, float *W_k, float *W_v, int embed_dim);

#endif
