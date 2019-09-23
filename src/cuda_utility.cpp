#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/version.h>

#include "utility.h"
#include "cuda_utility.h"

using std::wcout;
using std::endl;

GPUTimer::GPUTimer()
{
	cudaErrCheck(cudaEventCreate(&start));
	cudaErrCheck(cudaEventCreate(&end));
}

void GPUTimer::Start(cudaStream_t stream)
{
	if (started)
	{
		UPrintError("Timer already started.");
	}
	cudaErrCheck(cudaEventRecord(start));
	started = true;
	stopped = false;
	elapsedMS = -1.0f;
}

void GPUTimer::Stop(cudaStream_t stream)
{
	if (!started)
	{
		UPrintError("Timer has not been started.");
	}
	cudaErrCheck(cudaEventRecord(end));
	cudaErrCheck(cudaEventSynchronize(end));
	cudaErrCheck(cudaEventElapsedTime(&elapsedMS, start, end));
	started = false;
	stopped = true;
}

float GPUTimer::GetElapsedTime()
{
	if (!stopped)
	{
		UPrintError("Timer is still running.");
	}
	return elapsedMS;
}

void GPUTimer::PrintElapsedTime()
{
	if (!stopped)
	{
		UPrintError("Timer is still running.");
	}
	std::cout << "GPUTimer: Elapsed time in ms: " << elapsedMS << std::endl;
}

GPUTimer::~GPUTimer()
{
	cudaErrCheck(cudaEventDestroy(start));
	cudaErrCheck(cudaEventDestroy(end));
}

void CUDAUtil::PrintGPUSpecs()
{
	const int kb = 1024;
	const int mb = kb * kb;
	wcout << "NBody.GPU" << endl << "=========" << endl << endl;

	wcout << "CUDA version:   v" << CUDART_VERSION << endl;
	wcout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl << endl;

	int devCount;
	cudaGetDeviceCount(&devCount);
	wcout << "CUDA Devices: " << endl << endl;

	for (int i = 0; i < devCount; ++i)
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		wcout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
		wcout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
		wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
		wcout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
		wcout << "  Block registers: " << props.regsPerBlock << endl << endl;

		wcout << "  Warp size:         " << props.warpSize << endl;
		wcout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
		wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]" << endl;
		wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]" << endl;
		wcout << endl;
	}
}

void CUDAUtil::DEBUGSynchronizeDevice()
{
#ifdef _DEBUG
	cudaErrCheck(cudaPeekAtLastError()); cudaErrCheck(cudaDeviceSynchronize());
#endif
}