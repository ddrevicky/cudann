#pragma once

#include "e_matrix.h"
#include "gpu_matrix.h"

#define TEST_RANDOM_SEED 42

namespace TestUtil
{
	void RandomInit(EMatrix &m);
	bool IsApprox(EMatrix &e, GPUMatrix &g);
	bool AlmostEqual(double x, double y, double tolerance = 1e-5);
	double RelError(double x, double y, double tolerance = 1e-5);
	double RelError(EMatrix &e1, EMatrix &e2, double tolerance = 1e-5);
}
