#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                                                \
    {                                                                                                   \
        cudaError_t err = (call);                                                                       \
        if (err != cudaSuccess)                                                                         \
        {                                                                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                                                    \
        }                                                                                               \
    }

void *cuda_alloc(size_t size);
void cuda_free(void *ptr);
void copy_to_device(void *dst, const void *src, size_t size);
void copy_to_host(void *dst, const void *src, size_t size);

#endif