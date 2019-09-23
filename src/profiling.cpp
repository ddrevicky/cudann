#include <iostream>

#include "profiling.h"
#include "test_util.h"
#include "layers_gpu.h"
#include "e_matrix.h"
#include "gpu_matrix.h"

void EigenMatMul(EMatrix &X, EMatrix &Y, EMatrix &Z)
{
	Z = X * Y;
}

void Profiling::ProfileMatMul()
{
	std::cout << "TESTING SPEED\n";

	unsigned D1 = 256;
	unsigned D2 = 700;
	unsigned D3 = 1000;
	EMatrix X(D1, D2);
	EMatrix Y(D2, D3);
	TestUtil::RandomInit(X);
	TestUtil::RandomInit(Y);
	EMatrix result(D1, D3);

	GPUMatrix X_g(int(X.rows()), int(X.cols()));
	X_g.SetFromMem(X.data());
	GPUMatrix Y_g(int(Y.rows()), int(Y.cols()));
	Y_g.SetFromMem(Y.data());
	GPUMatrix result_g(int(result.rows()), int(result.cols()));

	ProfileLayerGPU::MatMul(X_g, Y_g, result_g);
	Util::ProfileFunction(10, EigenMatMul, "EigenMatMul", X, Y, result);

	X_g.Release();
	Y_g.Release();
	result_g.Release();
}

void Profiling::ProfileMatExp()
{
	EMatrix X(300, 580);
	TestUtil::RandomInit(X);

	EMatrix max(300, 1);
	TestUtil::RandomInit(max);

	EMatrix XExp(300, 580);

	E_TO_GPU_MATRIX(X, X_g);
	E_TO_GPU_MATRIX(max, max_g);
	E_TO_GPU_MATRIX(XExp, XExp_g);

	ProfileLayerGPU::MatExp(X_g, max_g, XExp_g);

	X_g.Release();
	max_g.Release();
	XExp_g.Release();
}