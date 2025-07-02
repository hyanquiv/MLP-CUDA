#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream> // Añadido para std::cerr

#define CHECK_CUDA(call)                                                  \
    {                                                                     \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess)                                           \
        {                                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
            std::cerr << " - " << cudaGetErrorString(err) << std::endl;   \
            throw std::runtime_error("CUDA error");                       \
        }                                                                 \
    }

void *cuda_alloc(size_t size);
void cuda_free(void *ptr);
void copy_to_device(void *dst, const void *src, size_t size);
void copy_to_host(void *dst, const void *src, size_t size);

#endif