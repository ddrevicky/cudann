#pragma once

#include <utility>

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "utility.h"

#define cudaErrCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess)
	{
#if defined(_DEBUG) || defined(DEBUG)
		fprintf(stderr, "cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
		assert(false);
#else
		fprintf(stderr, "CUDA failure: %s \n", cudaGetErrorString(code));
		cudaDeviceReset();
		exit(code);
#endif
	}
}

class GPUTimer
{
public:
	GPUTimer();
	~GPUTimer();
	void Start(cudaStream_t stream = 0);
	void Stop(cudaStream_t stream = 0);
	float GetElapsedTime();
	void PrintElapsedTime();
private:
	cudaEvent_t start;
	cudaEvent_t end;
	bool started = false;
	bool stopped = false;
	float elapsedMS = -1.0f;
};

namespace CUDAUtil
{
	void PrintGPUSpecs();
	void DEBUGSynchronizeDevice();
}
