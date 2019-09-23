#pragma once

namespace TestGPU
{
	// Layers 
	void TestMatScale();
	void TestMatSquare();
	void TestMatColSum();
	void TestMatMul();
	void TestMatTranspose();
	void TestMatMatAdd();
	void TestMatRowMax();
	void TestMatRowSum();
	void TestMatSubMaxExp();
	void TestAffineLayerForward();
	void TestAffineLayerBackward();
	void TestReLULayerForward();
	void TestReLULayerBackward();
	void TestL2Regularization();
	void TestCrossEntropyLossForward();
	void TestCrossEntropyLossBackward();

	// NeuralNet
	void TestNeuralNetInit();
	void TestNeuralNetLoss();

	void RunAllTests();
}
