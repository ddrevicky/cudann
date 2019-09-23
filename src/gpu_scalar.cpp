#include <cuda_runtime.h>
#include <cuda.h>

#include "gpu_scalar.h"
#include "cuda_utility.h"

GPUScalar::GPUScalar(unsigned flags)
{
	size_t bytes = sizeof(float);
	cudaErrCheck(cudaMallocManaged(&this->data, bytes));
	if (flags & GPU_SET_ZERO)
		cudaErrCheck(cudaMemset(data, 0, bytes));
}

void GPUScalar::Set(float value)
{
	size_t bytes = sizeof(float);
	cudaErrCheck(cudaMemcpy(data, &value, bytes, cudaMemcpyHostToDevice));
}

void GPUScalar::Release()
{
	if (data)
	{
		cudaErrCheck(cudaFree(data));
		data = nullptr;
	}
}