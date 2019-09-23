#pragma once

namespace TestCPU
{
	// Layers
	void TestAffineLayerForward();
	void TestAffineLayerBackward();
	void TestReLULayerForward();
	void TestReLULayerBackward();
	void TestCrossEntropyLossForward();
	void TestCrossEntropyLossBackward();
	void TestL2Regularization();

	// NN
	void TestNeuralNetLoss();

	// Solver
	void TestSolver();

	void RunAllTests();
}