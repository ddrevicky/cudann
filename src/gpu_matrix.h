#pragma once

#include "gpu_defines.h"

#define E_TO_GPU_MATRIX(X, X_g) \
	GPUMatrix X_g(int((X).rows()), int((X).cols())); \
	(X_g).SetFromMem((X).data()); \

#define GPU_TO_E_MATRIX(X_g, X_e) \
	Eigen::Map<EMatrix> X_e(X_g.data, X_g.rows, X_g.cols); \

class GPUMatrix
{
public:
	GPUMatrix();
	GPUMatrix(int rows, int cols, unsigned flags = 0);
	void SetFromMem(float *mem);
	void Print(const char *name);
	void CopyToHost(float *dst);
	void Release();

public:
	int rows = 0;
	int cols = 0;
	int realElementCount = 0;
	float *data = nullptr;
};