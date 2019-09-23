#include <cstring>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#include "cuda_utility.h"
#include "gpu_matrix.h"
#include "e_matrix.h"

GPUMatrix::GPUMatrix()
{
}

GPUMatrix::GPUMatrix(int rows, int cols, unsigned flags)
{ 
	this->rows = rows; 
	this->cols = cols; 
	this->realElementCount = rows * cols;
	size_t bytes = realElementCount * sizeof(float);
	cudaErrCheck(cudaMallocManaged(&this->data, bytes));
	if (flags & GPU_SET_ZERO)
		cudaErrCheck(cudaMemset(data, 0, bytes));
}

void GPUMatrix::SetFromMem(float *mem)
{
	size_t bytes = rows * cols * sizeof(float);
	cudaErrCheck(cudaMemcpy(data, mem, bytes, cudaMemcpyHostToDevice));
}

void GPUMatrix::CopyToHost(float *dst)
{
	size_t bytes = rows * cols * sizeof(float);
	cudaErrCheck(cudaMemcpy(dst, data, bytes, cudaMemcpyDeviceToHost));
}

void GPUMatrix::Print(const char *name)
{
	std::cout << name << "\n";
	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < cols; ++c)
		{
			std::cout << data[r * cols + c] << " ";
		}
		std::cout << "\n";
	}
	std::cout << std::endl;
}

void GPUMatrix::Release()
{
	if (data)
	{
		cudaErrCheck(cudaFree(data));
		data = nullptr;
	}
	rows = 0;
	cols = 0;
	realElementCount = 0;
}