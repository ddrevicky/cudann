#pragma once

#include <utility>

namespace CUDAUtil
{
    template <typename F, typename... Args>
	double ProfileFunction(int numRuns, F function, const char *outputString, dim3 gridDim, dim3 blockDim, size_t sharedMemBytes, Args&&... args)
	{
		GPUTimer timerGPU;

		double totalTime = 0.0;
		for (int i = 0; i < numRuns; ++i)
		{
			timerGPU.Start();
			function<<<gridDim, blockDim, sharedMemBytes>>>(std::forward<Args>(args)...);
			timerGPU.Stop();
			totalTime += double(timerGPU.GetElapsedTime());
			cudaErrCheck(cudaPeekAtLastError()); cudaErrCheck(cudaDeviceSynchronize());
		}

		double averageTime = totalTime / double(numRuns);
		printf("-------------------------------------------------\n");
		printf("GPU PROFILING %s, runs: %d \n", outputString, numRuns);
		printf("Average time %lf ms \n", averageTime);
		printf("-------------------------------------------------\n");
		return averageTime;
	}
}