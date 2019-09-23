#include <chrono>
#include <iostream>

#include "utility.h"

void Util::Error(const char *file, int line, const char *message)
{
	std::cerr << "ERROR in file " << file << ", line " << line << " MESSAGE: " << message << std::endl;
	assert(false);
}

void Util::Assert(const char *file, int line, bool condition, const char *message)
{
	if (!condition)
	{
		std::cerr << "Assert failed in file " << file << ", line " << line << " MESSAGE: " << message << std::endl;
		assert(false);
	}
}

void CPUTimer::Start()
{
	if (started)
	{
		UPrintError("CPUTimer already started.");
	}
	start = std::chrono::high_resolution_clock::now();
	double elapsedMS = 0.0;
	started = true;
	stopped = false;
}

void CPUTimer::Stop()
{
	if (!started)
	{
		UPrintError("CPUTimer has not been started.");
	}
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - start;
	elapsedMS = 1000 * diff.count();
	started = false;
	stopped = true;
}

double CPUTimer::GetElapsedTime()
{
	if (!stopped)
	{
		UPrintError("CPUTimer is still running.");
	}
	return elapsedMS;
}

void CPUTimer::PrintElapsedTime()
{
	if (!stopped)
	{
		UPrintError("CPUTimer is still running.");
	}
	std::cout << "CPUTimer: Elapsed time in ms: " << elapsedMS << std::endl;
}