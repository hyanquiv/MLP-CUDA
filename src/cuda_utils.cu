#include "cuda_utils.h"

void *cuda_alloc(size_t size)
{
    void *ptr;
    CHECK_CUDA(cudaMalloc(&ptr, size));
    return ptr;
}

void cuda_free(void *ptr)
{
    CHECK_CUDA(cudaFree(ptr));
}

void copy_to_device(void *dst, const void *src, size_t size)
{
    CHECK_CUDA(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void copy_to_host(void *dst, const void *src, size_t size)
{
    CHECK_CUDA(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}