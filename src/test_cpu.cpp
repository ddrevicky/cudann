#include <iostream>
#include <random>

#include "utility.h"
#include "test_cpu.h"
#include "test_util.h"
#include "e_matrix.h"
#include "layers_cpu.h"
#include "neural_net_cpu.h"

void TestCPU::TestCrossEntropyLossForward()
{
	// Test 1
	{
		EMatrix scores(2, 4);
		scores << 1.0f, 2.0f, 3.0f, -2,
			42.0f, 45.0f, 80.0f, 90;
		EMatrix y(2, 1);
		y << 0,
			2;
		EMatrix expScores(2, 4);
		EMatrix expScoresSums(2, 1);

		float loss;
		LayerCPU::CrossEntropyLossForward(scores, y, expScores, expScoresSums, loss);
		ASSERT_FLT_EQ(loss, 6.2060623f)
	}

	// Test 2
	{
		EMatrix scores(500, 600);
		std::srand(42);
		TestUtil::RandomInit(scores);
		EMatrix y(500, 1);
		for (int i = 0; i < 500; ++i)
			y(i, 0) = float(std::rand() % 500);

		EMatrix expScores(500, 600);
		EMatrix expScoresSums(500, 1);

		float loss;
		LayerCPU::CrossEntropyLossForward(scores, y, expScores, expScoresSums, loss);
		ASSERT_FLT_EQ(loss, 9.14651299f)
	}
}

void TestCPU::TestCrossEntropyLossBackward()
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

	EMatrix bottomGradScores(2, 4);
	LayerCPU::CrossEntropyLossBackward(bottomGradScores, expScores, expScoresSums, yCorrect);

	EMatrix expectedBottomGradScores(2, 4);
	expectedBottomGradScores << -4.55185592e-01f, 1.21818207e-01f, 3.31136197e-01f, 2.23117811e-03f,
								7.12549707e-22f, 1.43119436e-20f, -4.99977291e-01f, 4.99977291e-01f;
	assert(bottomGradScores.isApprox(expectedBottomGradScores));
}

void TestCPU::TestL2Regularization()
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

	std::vector<EMatrix> regWeights = { W1, W2, W3 };
	float l2Loss = 0.0f;
	LayerCPU::L2RegularizedLoss(l2Loss, regStrength, regWeights);
	
	ASSERT_FLT_EQ(l2Loss, expectedLoss);
}

void TestCPU::TestAffineLayerForward()
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
	LayerCPU::AffineLayerForward(X, W, b, out);

	EMatrix expected(2, 3);
	expected << 7.00430012f, 100.24240112f, 118.f,
			   -69.90129852f, -3574.56030273f, 3350.0f;

	assert(out.isApprox(expected));
}

void TestCPU::TestAffineLayerBackward()
{
	float regularizationStrength = 1.2f;

	EMatrix bottomA(3, 5);
	bottomA <<  1.0f, 2.0f, 3.0f, -2.0f, 5.0f,
				42.0f, 45.0f, 80.0f, 90.0f, 3.0f,
				12.0f, 32.0f, 45.0f, -12.0f, -15.32f;

	EMatrix bottomW(5, 3);
	bottomW <<  0.01f, 0.02f, 0.03f,
				0.01f, 0.02f, 0.03f,
				-0.01f, -0.02f, 0.03f,
				0.03f, 0.02f, 0.03f,
				0.01f, 0.02f, -0.03f;

	
	EMatrix topGrad(3, 3);
	topGrad << -0.22414061f, 0.11711007f, 0.10703056f,
				0.00265786f, 0.00119425f, -0.00385211f,
				0.01154367f, -0.32227772f,  0.31073409f;


	EMatrix bottomGradA(bottomA.rows(), bottomA.cols());
	EMatrix bottomGradW(bottomW.rows(), bottomW.cols());
	EMatrix bottomGradBias(1, 3);

	LayerCPU::AffineLayerBackward(bottomGradA, bottomGradW, bottomGradBias, bottomA, bottomW, topGrad, regularizationStrength);

	EMatrix expectedGradA(bottomA.rows(), bottomA.cols());
	expectedGradA << 3.31171183e-03f, 3.31171183e-03f, 3.11012147e-03f, - 1.17110042e-03f, -3.11012147e-03f,
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

	assert(bottomGradA.isApprox(expectedGradA));
	assert(bottomGradW.isApprox(expectedGradW));
	assert(bottomGradBias.isApprox(expectedGradBias));
}

void TestCPU::TestReLULayerForward()
{
	EMatrix X(2, 4);
	X << 0.0, 2.0, 3.0, -2,
		42.0, 45.0, 80.0, -90;
	EMatrix out(2, 4);
	LayerCPU::ActivationLayerForward(X, out, ActivationFunction::ReLU);

	EMatrix expected(2, 4);
	expected << 0.0, 2.0, 3.0, 0,
		42.0, 45.0, 80.0, 0;

	assert(out.isApprox(expected));
}

void TestCPU::TestReLULayerBackward()
{
	EMatrix X(2, 4);
	X << 0.0, 2.0, 3.0, -2,
		42.0, 45.0, 80.0, -90;
	EMatrix topGrad(2, 4);
	topGrad << 0.0, 12.0, 3.0, 30.0,
		0.0, -45.0, 800.0, 20.0;

	EMatrix bottomGrad(2, 4);
	LayerCPU::ActivationLayerBackward(bottomGrad, X, topGrad, ActivationFunction::ReLU);

	EMatrix expected(2, 4);
	expected << 0.0, 12.0, 3.0, 0.0,
		0.0, -45.0, 800.0, 0.0;

	assert(bottomGrad.isApprox(expected));
}

void TestCPU::TestNeuralNetLoss()
{
	{
		unsigned N = 3;
		unsigned D = 5;
		unsigned hidden = 3;
		unsigned C = 4;
		float regStrength = 1.2f;
		float weightScale = 0.001f;
		float learnRate = 0.01f;
		int randSeed = TEST_RANDOM_SEED;
		int batchSize = 512;
		unsigned int modelFlags = MODEL_PREDICT_LOSS | MODEL_PREDICT_SCORES | MODEL_GRAD_UPDATE;

		NeuralNetCPU neuralNet(std::vector<unsigned> {D, hidden, C}, ActivationFunction::ReLU, regStrength, weightScale, randSeed, batchSize);

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

		neuralNet.weights[0] = w0;
		neuralNet.weights[1] = w1;
		neuralNet.biases[0] = b0;
		neuralNet.biases[1] = b1;

		Prediction result = neuralNet.Loss(xTrain, yTrain, modelFlags, learnRate);
		ASSERT_FLT_EQ(result.loss, 1.328414885f);

		EMatrix expectedGradW0(D, hidden);
		expectedGradW0 <<
			-0.33143398f, 0.13725609f, - 0.18638609f,
			-0.35596496f, 0.14616545f, - 0.37273127f,
			-0.66615993f, 0.19269262f, - 0.58476698f,
			-0.69992989f, 0.2630347f, - 0.10470915f,
			-0.012531f, 0.03644016f, 0.09216708f;
		EMatrix expectedGradb0(1, hidden);
		expectedGradb0 << -0.008177f, 0.00355825f, -0.01165537f;

		EMatrix expectedGradW1(hidden, C);
		expectedGradW1 <<
			0.30670792f,  0.32787156f, - 0.58200014f,  0.0794207f,
			0.22121917f,  0.3660714f, - 0.39014077f, 0.0548502f,
			0.96758294f,  0.2374015f, - 1.46499705f, 0.45201254f;
		EMatrix expectedGradb1(1, C);
		expectedGradb1 << -0.07662527f, -0.05100203f, -0.06252852f,  0.1901558f;

		assert(neuralNet.gradWeights[0].isApprox(expectedGradW0));
		assert(neuralNet.gradBiases[0].isApprox(expectedGradb0));
		assert(neuralNet.gradWeights[1].isApprox(expectedGradW1));
		assert(neuralNet.gradBiases[1].isApprox(expectedGradb1));
	}

	{
		unsigned N = 3;
		unsigned D = 5;
		unsigned hidden1 = 3;
		unsigned hidden2 = 3;
		unsigned C = 4;
		float regStrength = 1.2f;
		float weightScale = 0.001f;
		float learnRate = 0.01f;
		int randSeed = TEST_RANDOM_SEED;
		int batchSize = 512;
		unsigned int modelFlags = MODEL_PREDICT_LOSS | MODEL_PREDICT_SCORES | MODEL_GRAD_UPDATE;

		NeuralNetCPU neuralNet(std::vector<unsigned> {D, hidden1, hidden2, C}, ReLU, regStrength, weightScale, randSeed, batchSize);

		EMatrix xTrain(N, D);
		xTrain << 1.0f, 2.0f, 3.0f, -2.0f, 5.0f,
			42.0f, 45.0f, 80.0f, 90.0f, 3.0f,
			12.0f, 32.0f, 45.0f, -12.0f, -15.32f;
		EMatrix yTrain(N, 1);
		yTrain << 0,
			2,
			1;

		EMatrix w0(D, hidden1);
		w0 << 0.01f, 0.02f, 0.03f,
			0.01f, 0.02f, 0.03f,
			-0.01f, -0.02f, 0.03f,
			0.03f, 0.02f, 0.03f,
			0.01f, 0.02f, -0.03f;
		EMatrix b0(1, hidden1);
		b0 << 0, 0, 0;

		EMatrix w1(hidden1, hidden2);
		w1 << 0.05f, 0.04f, 0.03f,
			  0.01f, 0.52f, 0.13f,
			-0.01f, -0.12f, 0.13f;
		EMatrix b1(1, hidden2);
		b1 << 0, 0, 0;

		EMatrix w2(hidden2, C);
		w2 << 0.05f, 0.02f, 0.06f, -0.02f,
			0.05f, 0.12f, 0.06f, -0.02f,
			0.05f, 0.07f, 0.06f, -0.02f;
		EMatrix b2(1, C);
		b2 << 0, 0, 0, 0;

		neuralNet.weights[0] = w0;
		neuralNet.biases[0] = b0;
		neuralNet.weights[1] = w1;
		neuralNet.biases[1] = b1;
		neuralNet.weights[2] = w2;
		neuralNet.biases[2] = b2;

		Prediction result = neuralNet.Loss(xTrain, yTrain, modelFlags, learnRate);
		ASSERT_FLT_EQ(result.loss, 1.5953909874f);

		EMatrix expectedGradW0(D, hidden1);
		expectedGradW0 <<
			-0.01997885f, - 0.04761392f,  0.00032119f,
			-0.02226305f, - 0.05278178f, - 0.02674211f,
			-0.07291209f, - 0.16046949f, - 0.06031633f,
			-0.03252609f, - 0.12922379f,  0.00784322f,
			0.0097158f,   0.01860563f, - 0.01783165f;
		EMatrix expectedGradb0(1, hidden1);
		expectedGradb0 << -0.0007614f, -0.00176037f, -0.00176423f;

		EMatrix expectedGradW1(hidden1, hidden2);
		expectedGradW1 <<
			0.03094991f,  0.0435697f,   0.0192598f,
			-0.00919943f,  0.62088734f,  0.14384393f,
			-0.09105776f, - 0.15605678f,  0.08315954f;
		EMatrix expectedGradb1(1, hidden2);
		expectedGradb1 << -0.01786446f, - 0.00071887f, -0.01914262f;

		EMatrix expectedGradW2(hidden2, C);
		expectedGradW2 <<
			0.06692371f,  0.03141847f,  0.05130851f, - 0.01765068f,
			0.07225626f,  0.16749772f,  0.0157922f, - 0.00354617f,
			0.20081569f,  0.11231138f, - 0.22746101f,  0.10633391f;
		EMatrix expectedGradb2(1, C);
		expectedGradb2 << -0.08196013f, -0.07764593f, -0.08021595f,  0.23982203f;

		assert(neuralNet.gradWeights[0].isApprox(expectedGradW0));
		assert(neuralNet.gradBiases[0].isApprox(expectedGradb0));
		assert(neuralNet.gradWeights[1].isApprox(expectedGradW1));
		assert(neuralNet.gradBiases[1].isApprox(expectedGradb1));
		assert(neuralNet.gradWeights[2].isApprox(expectedGradW2));
		assert(neuralNet.gradBiases[2].isApprox(expectedGradb2));
	}
}

void TestCPU::RunAllTests()
{
	std::cout << "Running CPU tests ...";

	// Layers CPU
	TestCPU::TestAffineLayerForward();
	TestCPU::TestAffineLayerBackward();
	TestCPU::TestReLULayerForward();
	TestCPU::TestReLULayerBackward();
	TestCPU::TestCrossEntropyLossForward();
	TestCPU::TestCrossEntropyLossBackward();
	TestCPU::TestL2Regularization();

	// NeuralNetCPU
	TestCPU::TestNeuralNetLoss();
	
	std::cout << " Passed.\n";
}