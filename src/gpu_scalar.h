#pragma once

#include "gpu_defines.h"

class GPUScalar
{
public:
	GPUScalar(unsigned flags = 0);
	void Set(float value);
	void Release();

public:
	float *data = nullptr;
};