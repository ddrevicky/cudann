#pragma once

#include <chrono>
#include <utility>
#include <cassert>

#define UPrintError(message) Util::Error(__FILE__, __LINE__, message)
#define UAssert(condition, message) Util::Assert(__FILE__, __LINE__, condition, message)

#define ASSERT_FLT_EQ(a, b) \
		assert(abs((a) - (b)) <= 0.0001 * std::max(abs(a), abs(b))); \

class CPUTimer
{
public:
	void Start();
	void Stop();
	double GetElapsedTime();
	void PrintElapsedTime();
private:
	bool started = false;
	bool stopped = false;
	std::chrono::time_point<std::chrono::high_resolution_clock>  start;
	std::chrono::time_point<std::chrono::high_resolution_clock> end;
	double elapsedMS = 0.0;
};

namespace Util
{
	void Error(const char *file, int line, const char *message);
	void Assert(const char *file, int line, bool condition, const char *message);
	
	template <typename F, typename... Args>
	double ProfileFunction(int numRuns, F function, const char *outputString, Args&&... args)
	{
		CPUTimer timerCPU;
		
		timerCPU.Start();
		for (int i = 0; i < numRuns; ++i)
		{
			function(std::forward<Args>(args)...);
		}
		timerCPU.Stop();

#ifdef _DEBUG
		printf("-------------------------------------------------\n");
		printf("CPU PROFILING %s, runs: %d \n", outputString, numRuns);
		printf("Average time %lf ms \n", averageTime);
		printf("-------------------------------------------------\n");
#endif
		double averageTime = timerCPU.GetElapsedTime() / double(numRuns);
		return averageTime;
	}
}