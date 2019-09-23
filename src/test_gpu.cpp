#include <random>
#include <iostream>

#include "utility.h"
#include "test_util.h"
#include "test_gpu.h"
#include "e_matrix.h"
#include "gpu_matrix.h"
#include "cuda_utility.h"
#include "layers_gpu.h"
#include "kernels_gpu.h"
#include "neural_net_cpu.h"
#include "neural_net_gpu.h"

#define DEFAULT_CUDA_STREAM 0 

void TestGPU::TestNeuralNetInit()
{
	unsigned N = 3;
	unsigned D = 5;
	unsigned hidden = 3;
	unsigned C = 4;
	float regStrength = 0;
	float weightScale = 0.0001f;
	int randSeed = TEST_RANDOM_SEED;
	bool deviceOnly = false;
	int batchSize = 256;

	NeuralNetCPU netCPU(std::vector<unsigned> {D, hidden, C}, ReLU, regStrength, weightScale, randSeed, batchSize);
	NeuralNetGPU netGPU(std::vector<unsigned> {D, hidden, C}, ReLU, regStrength, weightScale, randSeed, batchSize);

	for (size_t i = 0; i < netGPU.weights.size(); ++i)
	{
		Eigen::Map<EMatrix> gpuWeight(netGPU.weights[i].data, netCPU.weights[i].rows(), netCPU.weights[i].cols());
		assert(gpuWeight.isApprox(netCPU.weights[i]));

		Eigen::Map<EMatrix> gpuBias(netGPU.biases[i].data, netCPU.biases[i].rows(), netCPU.biases[i].cols());
		assert(gpuBias.isApprox(netCPU.biases[i]));

		assert(netGPU.gradWeights[i].rows == netCPU.gradWeights[i].rows());
		assert(netGPU.gradWeights[i].cols == netCPU.gradWeights[i].cols());

		assert(netGPU.gradBiases[i].rows == netCPU.gradBiases[i].rows());
		assert(netGPU.gradBiases[i].cols == netCPU.gradBiases[i].cols());
	}
}

void TestGPU::TestMatMul()
{
	// Test 1
	{
		EMatrix X(2, 2);
		X << 2.0f, 2.0f,
			2.0f, 2.0f;
		EMatrix W(2, 2);
		W << 1.0f, 1.0f,
			1.0f, 1.0f;

		EMatrix result = X * W;

		E_TO_GPU_MATRIX(X, X_g);
		E_TO_GPU_MATRIX(W, W_g);
		GPUMatrix result_g(int(result.rows()), int(result.cols()));

		OpGPU::MatMul(X_g, W_g, result_g, DEFAULT_CUDA_STREAM);
		assert(TestUtil::IsApprox(result, result_g));

		X_g.Release();
		W_g.Release();
		result_g.Release();
	}

	// Test 2
	{
		EMatrix X(2, 4);
		X << 1.0f, 2.0f, 3.0f, -2.0f,
			7.0f, 5.0f, 8.0f, 6.0f;
		EMatrix W(4, 3);
		W << 1.0f, -0.1f, 5.0f,
			2.0f, 0.1f, -2.0f,
			-1.0f, 0.1f, 2.0f,
			1.0f, -4.0f, 4.0f;

		EMatrix result = X * W;

		E_TO_GPU_MATRIX(X, X_g);
		E_TO_GPU_MATRIX(W, W_g);
		GPUMatrix result_g(int(result.rows()), int(result.cols()));

		OpGPU::MatMul(X_g, W_g, result_g, DEFAULT_CUDA_STREAM);
		assert(TestUtil::IsApprox(result, result_g));

		X_g.Release();
		W_g.Release();
		result_g.Release();
	}

	// Test 3
	{
		EMatrix X(731, 451);
		TestUtil::RandomInit(X);
		EMatrix W(451, 521);
		TestUtil::RandomInit(W);

		EMatrix result = X * W;

		E_TO_GPU_MATRIX(X, X_g);
		E_TO_GPU_MATRIX(W, W_g);
		GPUMatrix result_g(int(result.rows()), int(result.cols()));

		OpGPU::MatMul(X_g, W_g, result_g, DEFAULT_CUDA_STREAM);
		assert(TestUtil::IsApprox(result, result_g));

		X_g.Release();
		W_g.Release();
		result_g.Release();
	}
}

void TestGPU::TestMatTranspose()
{
	// Test 1
	{
		EMatrix W(4, 3);
		W << 0.0001f, -0.0001f, 25.0f,
			0.0021f, 0.121f, -2.0f,
			-1.0f, 0.0001f, 25.0f,
			0.0f, -40.0001f, 4.0f;

		EMatrix WTrans = W.transpose();

		E_TO_GPU_MATRIX(W, W_g);
		GPUMatrix WTrans_g(int(WTrans.rows()), int(WTrans.cols()));

		OpGPU::MatTranspose(W_g, WTrans_g, DEFAULT_CUDA_STREAM);
		assert(TestUtil::IsApprox(WTrans, WTrans_g));

		WTrans_g.Release();
	}

	// Test 2
	{
		EMatrix W(1233, 751);
		TestUtil::RandomInit(W);
		EMatrix WTrans = W.transpose();

		E_TO_GPU_MATRIX(W, W_g);
		GPUMatrix WTrans_g(int(WTrans.rows()), int(WTrans.cols()));

		OpGPU::MatTranspose(W_g, WTrans_g, DEFAULT_CUDA_STREAM);
		assert(TestUtil::IsApprox(WTrans, WTrans_g));

		WTrans_g.Release();
	}
}

void TestGPU::TestMatMatAdd()
{
	// Test 1
	{
		EMatrix X(4, 3);
		X << 0.0001f, -0.0001f, 25.0f,
			0.0021f, 0.121f, -2.0f,
			-1.0f, 0.0001f, 25.0f,
			0.0f, -40.0001f, 4.0f;

		EMatrix Y(4, 3);
		Y << 0.031f, -23.0001f, 125.0f,
			0.1011021f, 0.121f, -32.0f,
			-10.0f, 0.0001f, 255.0f,
			550.0f, -440.0001f, 14.0f;

		float beta = 42.0f;

		EMatrix Z = X + beta * Y;

		E_TO_GPU_MATRIX(X, X_g);
		E_TO_GPU_MATRIX(Y, Y_g);
		E_TO_GPU_MATRIX(Z, Z_g);

		OpGPU::MatMatAdd(X_g, Y_g, Z_g, beta, DEFAULT_CUDA_STREAM);

		assert(TestUtil::IsApprox(Z, Z_g));

		X_g.Release();
		Y_g.Release();
		Z_g.Release();
	}

	// Test 2
	{
		EMatrix X(1233, 751);
		TestUtil::RandomInit(X);

		EMatrix Y(1233, 751);
		TestUtil::RandomInit(Y);

		float beta = 42.0f;

		EMatrix Z = X + beta * Y;

		E_TO_GPU_MATRIX(X, X_g);
		E_TO_GPU_MATRIX(Y, Y_g);
		E_TO_GPU_MATRIX(Z, Z_g);

		OpGPU::MatMatAdd(X_g, Y_g, Z_g, beta, DEFAULT_CUDA_STREAM);
		assert(TestUtil::IsApprox(Z, Z_g));

		X_g.Release();
		Y_g.Release();
		Z_g.Release();
	}
}

void TestGPU::TestMatColSum()
{
	// Test 1
	{
		EMatrix X(2, 3);
		X << 1.0f, 2.0f, 42.0f,
			5.0f, 4.0f, -42.0f;
		EMatrix v(1, 3);
		v << 2.0f, 1.0f, 15.0f;

		EMatrix expected(1, 3);
		expected << 6.0f, 6.0f, 0.0f;

		E_TO_GPU_MATRIX(X, X_g);
		E_TO_GPU_MATRIX(v, v_g);

		OpGPU::MatColSum(X_g, v_g, DEFAULT_CUDA_STREAM);

		assert(TestUtil::IsApprox(expected, v_g));

		X_g.Release();
		v_g.Release();
	}

	// Test 2
	{
		EMatrix X(703, 903);
		TestUtil::RandomInit(X);

		EMatrix v(1, X.cols());
		TestUtil::RandomInit(v);

		EMatrix expected = EMatrix(v).setZero();
		for (int r = 0; r < X.rows(); ++r)
		{
			for (int c = 0; c < X.cols(); ++c)
			{
				expected(0, c) += X(r, c);
			}
		}

		E_TO_GPU_MATRIX(X, X_g);
		E_TO_GPU_MATRIX(v, v_g);

		OpGPU::MatColSum(X_g, v_g, DEFAULT_CUDA_STREAM);
		assert(TestUtil::IsApprox(expected, v_g));

		X_g.Release();
		v_g.Release();
	}
}

void TestGPU::TestMatRowMax()
{
	// Test 1
	{
		EMatrix X(3, 5);
		X << -1.0f, -2.0f, -3.0f, -2.0f, -5.0f,
			42.0f, 45.0f, 80.0f, 90.0f, 3.0f,
			12.0f, 32.0f, 45.0f, -12.0f, -15.32f;

		EMatrix max(3, 1);

		EMatrix expected(3, 1);
		expected << -1.0f,
			90.0f,
			45.0f;

		E_TO_GPU_MATRIX(X, X_g);
		E_TO_GPU_MATRIX(max, max_g);

		OpGPU::MatRowMax(X_g, max_g, DEFAULT_CUDA_STREAM);
		assert(TestUtil::IsApprox(expected, max_g));

		X_g.Release();
		max_g.Release();
	}

	// Test 2
	{
		EMatrix X(703, 903);
		TestUtil::RandomInit(X);

		EMatrix max(X.rows(), 1);

		EMatrix expected(X.rows(), 1);
		for (int r = 0; r < X.rows(); ++r)
		{
			EMatrix XRow = X.block(r, 0, 1, X.cols());
			expected(r, 0) = XRow.maxCoeff();
		}

		E_TO_GPU_MATRIX(X, X_g);
		E_TO_GPU_MATRIX(max, max_g);

		OpGPU::MatRowMax(X_g, max_g, DEFAULT_CUDA_STREAM);
		assert(TestUtil::IsApprox(expected, max_g));

		X_g.Release();
		max_g.Release();
	}
}

void TestGPU::TestMatRowSum()
{
	// Test 1
	{
		EMatrix X(3, 5);
		X << 1.0f, 2.0f, 3.0f, -2.0f, 5.0f,
			2.0f, 5.0f, -8.0f, 9.0f, 3.0f,
			1.0f, 2.0f, 4.0f, -12.0f, -1.0f;

		EMatrix xRowSum = EMatrix(3, 1).setZero();

		EMatrix expected(3, 1);
		expected << 9.0f,
					11.0f,
					-6.0f;

		E_TO_GPU_MATRIX(X, X_g);
		E_TO_GPU_MATRIX(xRowSum, xRowSum_g);

		OpGPU::MatRowSum(X_g, xRowSum_g, DEFAULT_CUDA_STREAM);
		assert(TestUtil::IsApprox(expected, xRowSum_g));

		X_g.Release();
		xRowSum_g.Release();
	}

	// Test 2
	{
		EMatrix X(1703, 933);
		TestUtil::RandomInit(X);

		EMatrix xRowSum = EMatrix(X.rows(), 1).setZero();

		EMatrix expected(X.rows(), 1);
		for (int r = 0; r < X.rows(); ++r)
		{
			EMatrix row = X.block(r, 0, 1, X.cols());
			expected(r, 0) = row.sum();
		}

		E_TO_GPU_MATRIX(X, X_g);
		E_TO_GPU_MATRIX(xRowSum, xRowSum_g);

		OpGPU::MatRowSum(X_g, xRowSum_g, DEFAULT_CUDA_STREAM);
		assert(TestUtil::IsApprox(expected, xRowSum_g));

		X_g.Release();
		xRowSum_g.Release();
	}
}

void TestGPU::TestMatSquare()
{
	// Test 1
	{
		EMatrix X(503, 703);
		TestUtil::RandomInit(X);

		EMatrix XSquare(X.rows(), X.cols());
		for (int r = 0; r < X.rows(); ++r)
		{
			for (int c = 0; c < X.cols(); ++c)
			{
				XSquare(r, c) = X(r, c) * X(r, c);
			}
		}

		E_TO_GPU_MATRIX(X, X_g);
		E_TO_GPU_MATRIX(XSquare, XSquare_g);

		OpGPU::MatSquare(X_g, XSquare_g, DEFAULT_CUDA_STREAM);
		assert(TestUtil::IsApprox(XSquare, XSquare_g));

		X_g.Release();
		XSquare_g.Release();
	}
}

void TestGPU::TestMatScale()
{
	// Test 1
	{
		EMatrix X(503, 703);
		TestUtil::RandomInit(X);

		E_TO_GPU_MATRIX(X, XScale_g);

		float scale = 123.0f;
		EMatrix XScale = X * scale;

		OpGPU::MatScale(XScale_g, scale, DEFAULT_CUDA_STREAM);
		assert(TestUtil::IsApprox(XScale, XScale_g));

		XScale_g.Release();
	}
}

void TestGPU::TestMatSubMaxExp()
{
	// Test 1
	{
		EMatrix X(3, 5);
		X << 1.0f, 2.0f, 3.0f, -2.0f, 5.0f,
			42.0f, 45.0f, 80.0f, 90.0f, 3.0f,
			12.0f, 32.0f, 45.0f, -12.0f, -15.32f;

		EMatrix xRowMax(3, 1);
		xRowMax << 5.0f,
			90.0f,
			45.0f;

		EMatrix XExp(3, 5);

		EMatrix expected(3, 5);
		expected << 1.83156389e-02f, 4.97870684e-02f, 1.35335283e-01f, 9.11881966e-04f, 1.00000000e+00f,
			1.42516408e-21f, 2.86251858e-20f, 4.53999298e-05f, 1.00000000e+00f, 1.64581143e-38f,
			4.65888615e-15f, 2.26032941e-06f, 1.00000000e+00f, 1.75879220e-25f, 6.35853186e-27f;

		E_TO_GPU_MATRIX(X, X_g);
		E_TO_GPU_MATRIX(xRowMax, xRowMax_g);
		E_TO_GPU_MATRIX(XExp, XExp_g);

		OpGPU::MatSubMaxExp(X_g, xRowMax_g, XExp_g, DEFAULT_CUDA_STREAM);

		assert(TestUtil::IsApprox(expected, XExp_g));

		X_g.Release();
		xRowMax_g.Release();
		XExp_g.Release();
	}

	// Test 2
	{
		EMatrix X(503, 703);
		TestUtil::RandomInit(X);

		EMatrix xRowMax(X.rows(), 1);
		for (int r = 0; r < X.rows(); ++r)
		{
			EMatrix XRow = X.block(r, 0, 1, X.cols());
			xRowMax(r, 0) = XRow.maxCoeff();
		}

		EMatrix XExp(X.rows(), X.cols());
		for (int r = 0; r < X.rows(); ++r)
		{
			for (int c = 0; c < X.cols(); ++c)
			{
				XExp(r, c) = expf(X(r, c) - xRowMax(r, 0));
			}
		}

		E_TO_GPU_MATRIX(X, X_g);
		E_TO_GPU_MATRIX(xRowMax, xRowMax_g);
		E_TO_GPU_MATRIX(XExp, XExp_g);

		OpGPU::MatSubMaxExp(X_g, xRowMax_g, XExp_g, DEFAULT_CUDA_STREAM);

		assert(TestUtil::IsApprox(XExp, XExp_g));

		X_g.Release();
		xRowMax_g.Release();
		XExp_g.Release();
	}
}

void TestGPU::TestCrossEntropyLossForward()
{
	// Test 1
	{
		EMatrix scores(2, 4);
		scores << 1.0f, 2.0f, 3.0f, -2,
				  42.0f, 45.0f, 80.0f, 90;
		EMatrix yCorrect(2, 1);
		yCorrect << 0,
					2;

		E_TO_GPU_MATRIX(yCorrect, yCorrect_g);
		E_TO_GPU_MATRIX(scores, scores_g);
		GPUMatrix rowMax_g(int(yCorrect.rows()), int(yCorrect.cols()));
		GPUMatrix exp_g(int(scores.rows()), int(scores.cols()));
		GPUMatrix expSum_g(int(yCorrect.rows()), int(yCorrect.cols()));

		float *loss = nullptr;
		cudaErrCheck(cudaMallocManaged(&loss, sizeof(float)));
		*loss = 0.0f;

		LayerGPU::CrossEntropyLossForward(scores_g, yCorrect_g, rowMax_g, exp_g, expSum_g, loss, DEFAULT_CUDA_STREAM);

		ASSERT_FLT_EQ(*loss, 6.2060623f)

		cudaErrCheck(cudaFree(loss));
		yCorrect_g.Release();
		scores_g.Release();
		rowMax_g.Release();
		exp_g.Release();
		expSum_g.Release();
	}
	
	// Test 2
	{
		// Correct value for this test is based on the corresponding CPU test result. Modifying any of the values here
		// will result in the test being assessed as failed.
		EMatrix scores(500, 600);
		std::srand(42);
		TestUtil::RandomInit(scores);
		EMatrix yCorrect(500, 1);
		for (int i = 0; i < 500; ++i)
			yCorrect(i, 0) = float(std::rand() % 500);

		EMatrix expScores(500, 600);
		EMatrix expScoresSums(500, 1);

		E_TO_GPU_MATRIX(yCorrect, yCorrect_g);
		E_TO_GPU_MATRIX(scores, scores_g);
		E_TO_GPU_MATRIX(expScores, exp_g);
		E_TO_GPU_MATRIX(expScoresSums, expSum_g);
		GPUMatrix rowMax_g(int(yCorrect.rows()), int(yCorrect.cols()));

		float *loss = nullptr;
		cudaErrCheck(cudaMallocManaged(&loss, sizeof(float)));
		*loss = 0.0f;

		// Computes the CPU version which should be correct
		float CPUloss;
		LayerCPU::CrossEntropyLossForward(scores, yCorrect, expScores, expScoresSums, CPUloss);

		LayerGPU::CrossEntropyLossForward(scores_g, yCorrect_g, rowMax_g, exp_g, expSum_g, loss, DEFAULT_CUDA_STREAM);
		assert(TestUtil::IsApprox(scores, scores_g));
		assert(TestUtil::IsApprox(yCorrect, yCorrect_g));
		assert(TestUtil::IsApprox(expScores, exp_g));
		assert(TestUtil::IsApprox(expScoresSums, expSum_g));

		ASSERT_FLT_EQ(*loss, 9.14651299f)

		cudaErrCheck(cudaFree(loss));
		yCorrect_g.Release();
		scores_g.Release();
		rowMax_g.Release();
		exp_g.Release();
		expSum_g.Release();
	}
}

void TestGPU::TestCrossEntropyLossBackward()
{
	// Test 1
	{
		EMatrix yCorrect(2, 1);
		yCorrect << 0,
			2;

		EMatrix expScores(2, 4);
		expScores << 0.135335f, 0.367879f, 1.0f, 0.00673795f,
			1.42516e-21f, 2.86252e-20f, 4.53999e-05f, 1.0f;

		EMatrix expScoresSums(2, 1);
		expScoresSums << 1.50995f,
			1.00005f;

		E_TO_GPU_MATRIX(expScores, expScores_g);
		E_TO_GPU_MATRIX(expScoresSums, expScoresSums_g);
		E_TO_GPU_MATRIX(yCorrect, yCorrect_g);
		GPUMatrix gradScores_g(expScores_g.rows, expScores_g.cols);

		LayerGPU::CrossEntropyLossBackward(expScores_g, expScoresSums_g, yCorrect_g, gradScores_g, DEFAULT_CUDA_STREAM);

		EMatrix expectedGradScores(2, 4);
		expectedGradScores << -4.55185592e-01f, 1.21818207e-01f, 3.31136197e-01f, 2.23117811e-03f,
			7.12549707e-22f, 1.43119436e-20f, -4.99977291e-01f, 4.99977291e-01f;
		assert(TestUtil::IsApprox(expectedGradScores, gradScores_g));

		gradScores_g.Release();
		expScores_g.Release();
		expScoresSums_g.Release();
		yCorrect_g.Release();
	}

	// Test 2
	// Assumes the CPU version is correct
	{
		int D = 201;
		int N = 701;

		EMatrix yCorrect(N, 1);
		for (long r = 0; r < yCorrect.rows(); ++r)
			yCorrect(r, 0) = float(std::rand() % D);

		EMatrix expScores(N, D);
		TestUtil::RandomInit(expScores);

		EMatrix expScoresSums(N, 1);
		for (int r = 0; r < expScores.rows(); ++r)
		{
			EMatrix expScoresRow = expScores.block(r, 0, 1, expScores.cols());
			expScoresSums(r, 0) = expScoresRow.sum();
		}

		EMatrix bottomGradScores(N, D);

		E_TO_GPU_MATRIX(expScores, expScores_g);
		E_TO_GPU_MATRIX(expScoresSums, expScoresSums_g);
		E_TO_GPU_MATRIX(yCorrect, yCorrect_g);
		E_TO_GPU_MATRIX(bottomGradScores, bottomGradScores_g);

		for (int r = 0; r < bottomGradScores.rows(); ++r)
		{
			for (int c = 0; c < bottomGradScores.cols(); ++c)
			{
				bottomGradScores(r, c) = expScores(r, c) / expScoresSums(r, 0);
			}
			bottomGradScores(r, int(yCorrect(r, 0))) += -1.0;
		}
		bottomGradScores = bottomGradScores.array() / bottomGradScores.rows();

		LayerGPU::CrossEntropyLossBackward(expScores_g, expScoresSums_g, yCorrect_g, bottomGradScores_g, DEFAULT_CUDA_STREAM);
		assert(TestUtil::IsApprox(bottomGradScores, bottomGradScores_g));

		bottomGradScores_g.Release();
		expScores_g.Release();
		expScoresSums_g.Release();
		yCorrect_g.Release();
	}
}

void TestGPU::TestL2Regularization()
{
	EMatrix W1(20, 50);
	EMatrix W2(10, 10);
	EMatrix W3(10, 20);
	TestUtil::RandomInit(W1);
	TestUtil::RandomInit(W2);
	TestUtil::RandomInit(W3);

	float regStrength = 1.5f;
	float expectedLoss = W1.array().pow(2.0f).sum() + W2.array().pow(2.0f).sum() + W3.array().pow(2.0f).sum();
	expectedLoss *= 0.5f * regStrength;

	std::vector<GPUMatrix> regWeights_g;
	E_TO_GPU_MATRIX(W1, W1_g);
	E_TO_GPU_MATRIX(W2, W2_g);
	E_TO_GPU_MATRIX(W3, W3_g);

	regWeights_g.push_back(W1_g);
	regWeights_g.push_back(W2_g);
	regWeights_g.push_back(W3_g);

	float *l2Loss = nullptr;
	cudaErrCheck(cudaMallocManaged(&l2Loss, sizeof(float)));
	*l2Loss = 0.0f;

	GPUMatrix tmpMat_g(W1_g.rows, W1_g.cols);
	GPUMatrix tmpVec_g(W1_g.rows, 1);

	LayerGPU::L2RegularizedLoss(regWeights_g, l2Loss, regStrength, tmpMat_g, tmpVec_g, DEFAULT_CUDA_STREAM);

	ASSERT_FLT_EQ(*l2Loss, expectedLoss);

	cudaErrCheck(cudaFree(l2Loss));
	tmpMat_g.Release();
	tmpVec_g.Release();
	for (auto w : regWeights_g)
		w.Release();
}

void TestGPU::TestNeuralNetLoss()
{
	// Test 1
	{
		unsigned N = 3;
		unsigned D = 5;
		unsigned hidden = 3;
		unsigned C = 4;

		EMatrix xTrain(N, D);
		xTrain << 1.0f, 2.0f, 3.0f, -2.0f, 5.0f,
			42.0f, 45.0f, 80.0f, 90.0f, 3.0f,
			12.0f, 32.0f, 45.0f, -12.0f, -15.32f;
		EMatrix yTrain(N, 1);
		yTrain << 0,
			2,
			1;

		EMatrix w0(D, hidden);
		w0 << 0.01f, 0.02f, 0.03f,
			0.01f, 0.02f, 0.03f,
			-0.01f, -0.02f, 0.03f,
			0.03f, 0.02f, 0.03f,
			0.01f, 0.02f, -0.03f;
		EMatrix b0(1, hidden);
		b0 << 0, 0, 0;

		EMatrix w1(hidden, C);
		w1 << 0.05f, 0.02f, 0.06f, -0.02f,
			0.05f, 0.12f, 0.06f, -0.02f,
			0.05f, 0.07f, 0.06f, -0.02f;
		EMatrix b1(1, C);
		b1 << 0, 0, 0, 0;

		// Net
		float regStrength = 1.2f;
		float weightScale = 0.001f;
		int randSeed = TEST_RANDOM_SEED;
		unsigned int flags = MODEL_GRAD_UPDATE | MODEL_PREDICT_LOSS | MODEL_PREDICT_SCORES;
		int batchSize = N;
		NeuralNetGPU net(std::vector<unsigned> {D, hidden, C}, ReLU, regStrength, weightScale, randSeed, batchSize);
		net.weights[0].SetFromMem(w0.data());
		net.weights[1].SetFromMem(w1.data());
		net.biases[0].SetFromMem(b0.data());
		net.biases[1].SetFromMem(b1.data());

		E_TO_GPU_MATRIX(xTrain, xTrain_g);
		E_TO_GPU_MATRIX(yTrain, yTrain_g);

		Prediction result = net.Loss(&xTrain_g, &yTrain_g, flags, 0.0f);
		ASSERT_FLT_EQ(result.loss, 1.328414885f);

		EMatrix expectedGradW0(D, hidden);
		expectedGradW0 <<
			-0.33143398f, 0.13725609f, -0.18638609f,
			-0.35596496f, 0.14616545f, -0.37273127f,
			-0.66615993f, 0.19269262f, -0.58476698f,
			-0.69992989f, 0.2630347f, -0.10470915f,
			-0.012531f, 0.03644016f, 0.09216708f;
		EMatrix expectedGradb0(1, hidden);
		expectedGradb0 << -0.008177f, 0.00355825f, -0.01165537f;

		EMatrix expectedGradW1(hidden, C);
		expectedGradW1 <<
			0.30670792f, 0.32787156f, -0.58200014f, 0.0794207f,
			0.22121917f, 0.3660714f, -0.39014077f, 0.0548502f,
			0.96758294f, 0.2374015f, -1.46499705f, 0.45201254f;
		EMatrix expectedGradb1(1, C);
		expectedGradb1 << -0.07662527f, -0.05100203f, -0.06252852f, 0.1901558f;

		assert(TestUtil::IsApprox(expectedGradW0, net.gradWeights[0]));
		assert(TestUtil::IsApprox(expectedGradb0, net.gradBiases[0]));
		assert(TestUtil::IsApprox(expectedGradW1, net.gradWeights[1]));
		assert(TestUtil::IsApprox(expectedGradb1, net.gradBiases[1]));
	}

	// Test 2
	{
		unsigned N = 200;
		unsigned D = 700;
		unsigned hidden1 = 300;
		unsigned hidden2 = 200;
		unsigned C = 50;
		float regStrength = 1.2f;
		float weightScale = 0.001f;
		int randomSeed = TEST_RANDOM_SEED;
		float learnRate = 0.01f;
		int batchSize = N;

		NeuralNetCPU netCPU(std::vector<unsigned> {D, hidden1, hidden2, C}, ReLU, regStrength, weightScale, randomSeed, batchSize);
		NeuralNetGPU netGPU(std::vector<unsigned> {D, hidden1, hidden2, C}, ReLU, regStrength, weightScale, randomSeed, batchSize);

		EMatrix xTrain(N, D);
		TestUtil::RandomInit(xTrain);
		
		EMatrix yTrain(N, 1);
		TestUtil::RandomInit(yTrain);

		unsigned int flags = MODEL_GRAD_UPDATE | MODEL_PREDICT_LOSS | MODEL_PREDICT_SCORES;
		Prediction resultCPU = netCPU.Loss(xTrain, yTrain, flags, learnRate);

		E_TO_GPU_MATRIX(xTrain, xTrain_g);
		E_TO_GPU_MATRIX(yTrain, yTrain_g);
		Prediction resultGPU = netGPU.Loss(&xTrain_g, &yTrain_g, flags, learnRate);

		ASSERT_FLT_EQ(resultCPU.loss, resultGPU.loss);

		for (size_t i = 0; i < netCPU.gradBiases.size(); ++i)
		{
			EMatrix &dbCPU = netCPU.gradBiases[i];
			EMatrix dbGPU = Eigen::Map<EMatrix>(netGPU.gradBiases[i].data, netGPU.gradBiases[i].rows, netGPU.gradBiases[i].cols);
			double relError = TestUtil::RelError(dbCPU, dbGPU);
			assert(relError < 1e-1);

			EMatrix &dWCPU = netCPU.gradWeights[i];
			EMatrix dWGPU = Eigen::Map<EMatrix>(netGPU.gradWeights[i].data, netGPU.gradWeights[i].rows, netGPU.gradWeights[i].cols);
			relError = TestUtil::RelError(dWCPU, dWGPU);
			assert(relError < 1e-1);
		}
	}
}

void TestGPU::TestAffineLayerForward()
{
	EMatrix X(2, 4);
	X << 1.0f, 2.0f, 3.0f, -2,
		42.0f, 45.0f, 80.0f, 90.0f;
	EMatrix W(4, 3);
	W << 0.0001f, -0.0001f, 25.0f,
		0.0021f, 0.121f, -2.0f,
		-1.0f, 0.0001f, 25.0f,
		0.0f, -40.0001f, 4.0f;
	EMatrix b(1, 3);
	b << 10.0f, 20.0f, 30.0f;

	EMatrix out(2, 3);

	E_TO_GPU_MATRIX(X, X_g);
	E_TO_GPU_MATRIX(W, W_g);
	E_TO_GPU_MATRIX(b, b_g);
	GPUMatrix out_g(int(out.rows()), int(out.cols()));

	LayerGPU::AffineLayerForward(X_g, W_g, b_g, out_g, DEFAULT_CUDA_STREAM);

	EMatrix expected(2, 3);
	expected << 7.00430012f, 100.24240112f, 118.f,
		-69.90129852f, -3574.56030273f, 3350.0f;

	assert(TestUtil::IsApprox(expected, out_g));

	X_g.Release();
	W_g.Release();
	b_g.Release();
	out_g.Release();
}

void TestGPU::TestAffineLayerBackward()
{
	float regularizationStrength = 1.2f;

	EMatrix bottomA(3, 5);
	bottomA << 1.0f, 2.0f, 3.0f, -2.0f, 5.0f,
		42.0f, 45.0f, 80.0f, 90.0f, 3.0f,
		12.0f, 32.0f, 45.0f, -12.0f, -15.32f;

	EMatrix bottomW(5, 3);
	bottomW << 0.01f, 0.02f, 0.03f,
		0.01f, 0.02f, 0.03f,
		-0.01f, -0.02f, 0.03f,
		0.03f, 0.02f, 0.03f,
		0.01f, 0.02f, -0.03f;

	EMatrix topGrad(3, 3);
	topGrad << -0.22414061f, 0.11711007f, 0.10703056f,
		0.00265786f, 0.00119425f, -0.00385211f,
		0.01154367f, -0.32227772f, 0.31073409f;

	EMatrix bottomGradA(bottomA.rows(), bottomA.cols());
	EMatrix bottomGradW(bottomW.rows(), bottomW.cols());
	EMatrix bottomGradBias(1, 3);
	EMatrix bottomATrans = bottomA.transpose();
	EMatrix bottomWTrans = bottomW.transpose();

	E_TO_GPU_MATRIX(bottomA, bottomA_g);
	E_TO_GPU_MATRIX(bottomW, bottomW_g);
	E_TO_GPU_MATRIX(topGrad, topGrad_g);
	E_TO_GPU_MATRIX(bottomGradA, bottomGradA_g);
	E_TO_GPU_MATRIX(bottomGradW, bottomGradW_g);
	E_TO_GPU_MATRIX(bottomGradBias, bottomGradBias_g);
	E_TO_GPU_MATRIX(bottomATrans, bottomATrans_g);
	E_TO_GPU_MATRIX(bottomWTrans, bottomWTrans_g);

	LayerGPU::AffineLayerBackward(bottomGradA_g, bottomGradW_g, bottomGradBias_g, bottomA_g, bottomATrans_g, bottomW_g, bottomWTrans_g,
		topGrad_g, regularizationStrength, DEFAULT_CUDA_STREAM);

	EMatrix expectedGradA(bottomA.rows(), bottomA.cols());
	expectedGradA << 3.31171183e-03f, 3.31171183e-03f, 3.11012147e-03f, -1.17110042e-03f, -3.11012147e-03f,
		-6.50996881e-05f, -6.50996881e-05f, -1.66026846e-04f, -1.19425749e-05f, 1.66026846e-04f,
		2.99190497e-03f, 2.99190497e-03f, 1.56521406e-02f, 3.22277844e-03f, -1.56521406e-02f;

	EMatrix expectedGradW(bottomW.rows(), bottomW.cols());
	expectedGradW << 0.03801338f, -3.67606425f, 3.71005106f,
		0.05271978f, -10.00092602f, 10.02020741f,
		0.04767185f, -14.07962799f, 14.03195763f,
		0.58496416f, 3.76459503f, -4.25356054f,
		-1.27757859f, 5.55042791f, -4.27284956f;

	EMatrix expectedGradBias(1, 3);
	expectedGradBias << -0.20993908f, -0.2039734f, 0.41391253f;

	assert(TestUtil::IsApprox(expectedGradA, bottomGradA_g));
	assert(TestUtil::IsApprox(expectedGradW, bottomGradW_g));
	assert(TestUtil::IsApprox(expectedGradBias, bottomGradBias_g));

	bottomA_g.Release();
	bottomW_g.Release();
	topGrad_g.Release();
	bottomGradA_g.Release();
	bottomGradW_g.Release();
	bottomGradBias_g.Release();
	bottomATrans_g.Release();
	bottomWTrans_g.Release();
}

void TestGPU::TestReLULayerForward()
{
	EMatrix X(2, 4);
	X << 0.0, 2.0, 3.0, -2,
		42.0, 45.0, 80.0, -90;
	EMatrix out(2, 4);

	GPUMatrix X_g(int(X.rows()), int(X.cols()));
	X_g.SetFromMem(X.data());
	GPUMatrix out_g(int(out.rows()), int(out.cols()));

	LayerGPU::ActivationLayerForward(X_g, out_g, ReLU, DEFAULT_CUDA_STREAM);

	EMatrix expected(2, 4);
	expected << 0.0, 2.0, 3.0, 0,
		42.0, 45.0, 80.0, 0;

	assert(TestUtil::IsApprox(expected, out_g));

	X_g.Release();
	out_g.Release();
}

void TestGPU::TestReLULayerBackward()
{
	EMatrix X(2, 4);
	X << 0.0, 2.0, 3.0, -2,
		42.0, 45.0, 80.0, -90;
	EMatrix topGrad(2, 4);
	topGrad << 0.0, 12.0, 3.0, 30.0,
		0.0, -45.0, 800.0, 20.0;

	EMatrix bottomGrad(2, 4);

	GPUMatrix X_g(int(X.rows()), int(X.cols()));
	X_g.SetFromMem(X.data());
	GPUMatrix topGrad_g(int(topGrad.rows()), int(topGrad.cols()));
	topGrad_g.SetFromMem(topGrad.data());
	GPUMatrix bottomGrad_g(int(bottomGrad.rows()), int(bottomGrad.cols()));

	LayerGPU::ActivationLayerBackward(bottomGrad_g, X_g, topGrad_g, ReLU, DEFAULT_CUDA_STREAM);

	EMatrix expected(2, 4);
	expected << 0.0, 12.0, 3.0, 0.0,
		0.0, -45.0, 800.0, 0.0;

	assert(TestUtil::IsApprox(expected, bottomGrad_g));

	X_g.Release();
	topGrad_g.Release();
	bottomGrad_g.Release();
}

void TestGPU::RunAllTests()
{
	std::cout << "Running GPU tests...";

	// Layers GPU
	TestGPU::TestMatScale();
	TestGPU::TestMatSquare();
	TestGPU::TestMatColSum();
	TestGPU::TestMatMul();
	TestGPU::TestMatTranspose();
	TestGPU::TestMatMatAdd();
	TestGPU::TestMatRowMax();
	TestGPU::TestMatRowSum();
	TestGPU::TestMatSubMaxExp();
	TestGPU::TestAffineLayerForward();
	TestGPU::TestAffineLayerBackward();
	TestGPU::TestReLULayerForward();
	TestGPU::TestReLULayerBackward();
	TestGPU::TestL2Regularization();
	TestGPU::TestCrossEntropyLossForward();
	TestGPU::TestCrossEntropyLossBackward();

	// NeuralNetGPU
	TestGPU::TestNeuralNetInit();
	TestGPU::TestNeuralNetLoss();

	std::cout << " Passed.\n";
}